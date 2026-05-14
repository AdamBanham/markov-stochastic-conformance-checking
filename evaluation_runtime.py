import markov
from typing import List, Dict, Tuple, Callable, Set
from functools import wraps


class Reporter:
    """
    Tracks the runtimes for function calls. Users can set the context for
    the stored runtimes as desired inbetween calls.
    """

    def __init__(self) -> None:
        self._curr_path = []
        self._contexts = {}
        self._known_functions: Set[str] = set()

    def shift_root(self, context: str):
        """
        Shifts the root context for storing runtimes.
        """
        self._curr_path = [context]

    def shift(self, context: str):
        """
        Shifts the current context to a sub-context for storing runtimes.
        """
        self._curr_path.append(context)

    def shift_to_subpath(self, path: List[str]):
        """
        Shifts the context to the desired path for storing runtimes.
        """
        self.shift_root(path[0])

        for subpath in path[1:]:
            self.shift(subpath)

    def get_context_dict(self) -> Dict[str, List]:
        """
        Gets the current context dictionary for storing values.
        """
        curr_context = self._contexts
        for path in self._curr_path:
            if path not in curr_context:
                curr_context[path] = {
                    "_value": dict((func, []) for func in self._known_functions)
                }
            curr_context = curr_context[path]
        return curr_context["_value"]

    def add_watcher_to(self, func: Callable) -> Callable:
        """
        Adds a watcher to the runtime of the given callable.
        """
        self._known_functions.add(func.__name__)

        @wraps(func)
        def wrapped(*args, **kwargs):
            start = time()
            ret = func(*args, **kwargs)
            self.store_reported(func.__name__, time() - start)
            return ret

        return wrapped

    def add_custom_watcher_to(self, func: Callable, report: Callable) -> Callable:
        """
        Adds a custom watcher that calls report on the returned value of func
        and stores the returned value by report.
        """
        self._known_functions.add(func.__name__)

        @wraps(func)
        def wrapped(*args, **kwargs):
            ret = func(*args, **kwargs)
            self.store_reported(func.__name__, report(ret))
            return ret

        return wrapped

    def store_reported(self, func_name: str, reported: object):
        """
        Stores the runtime of the given
        """
        print(f"storing ({'/'.join(self._curr_path)}) :: {func_name} -> {reported}")
        storage = self.get_context_dict()
        storage[func_name].append(reported)

    def reset(self):
        """
        Resets storage.
        """
        self._contexts = {}


def collect_states(intersect: markov.IntersectedFiniteLabelledMarkovChain) -> int:
    return len(intersect._states)


reporter = Reporter()
markov.intersect_nets = reporter.add_custom_watcher_to(markov.intersect_nets, collect_states)

from measure import (
    compute_stochastic_entropy_recall,
    compute_stochastic_entropy_precision,
)
from pmkoalas.read import read_xes_simple
from pmkoalas.simple import EventLog, Trace
from pmkoalas._logging import info

from os import PathLike, mkdir


from os.path import join, exists
from random import Random
from json import dumps
from string import ascii_uppercase
from time import time
from itertools import product
from sys import set_int_max_str_digits
from copy import deepcopy

set_int_max_str_digits(18000)

LOG_FOLDER = join(".", "logs")
EVAL_LOGS = [
    # join(LOG_FOLDER, "road_fines.xes"),
    # join(LOG_FOLDER, "sepsis.xes"),
    join(LOG_FOLDER, "bpic_2020_permits.xes"),
]


compute_stochastic_entropy_precision = reporter.add_watcher_to(
    compute_stochastic_entropy_precision
)
compute_stochastic_entropy_recall = reporter.add_watcher_to(
    compute_stochastic_entropy_recall
)


def simplify_log_names(log: EventLog) -> Tuple[EventLog, Dict[str, str]]:
    # swap out long activity names
    info("imputing activity names...")
    swaps = {}
    traces = []
    options = [l for l in ascii_uppercase] + [
        l + r for l, r in product(ascii_uppercase, ascii_uppercase)
    ]
    for trace, freq in log:
        new_seq = []
        for act in trace:
            if act not in swaps:
                letter = options[0]
                options = options[1:]
                swaps[act] = letter
            new_seq += [swaps[act]]
        traces += [Trace(new_seq)] * freq

    # rebuild log
    info("rebuilding log...")
    log_name = log.get_name().lower().replace(" ", "_")
    return EventLog(traces, log_name), swaps


def sampler(log: EventLog, size: int, randomer: Random = None) -> EventLog:
    """
    Samples the log drawing up to size executions for the new sample log.
    Draws down the real traces into a dummy trace to make a strictly worse
    log in comparision to the real log.
    """
    traces = []
    pool = [(trace, count) for trace, count in log]
    pool = sorted(pool, key=lambda p: str(p[0]))
    pool_size = set(range(len(pool)))

    # dummy trace
    dummy = Trace(["dummy"])
    pool.append((dummy, 0))

    # pick traces
    for _ in range(size):
        pick = int(randomer.choice(list(pool_size)))
        trace, count = pool[pick]
        if count > 0:
            pool[pick] = (trace, max(0, count - 1))
            _, count = pool[-1]
            pool[-1] = (dummy, count + 1)
        else:
            print(
                f"picked a trace that did not have a non-zero count :: {trace, count}"
            )

        _, count = pool[pick]
        if count < 1:
            pool_size.remove(pick)

        if len(pool_size) < 1:
            break

    # build traces
    for trace, count in pool:
        for _ in range(count):
            traces.append(trace)

    # build sample log
    ret = EventLog(traces)

    if len(ret) != len(log):
        info(f"sample log was not the same size! :: {len(ret)=} vs {len(log)=}")

    return ret


def sampling(log: EventLog, n_samples: int, dump_directory: PathLike) -> Dict:
    """
    Computes a sample curve for the given log and number of samples.
    Each shifts a portion 1/n * |L| into a dummy trace to create a strictly
    worse stochastic distribution when compared to the source log.
    These samples drawn without replacement and draw deterministically
    within calls.

    Returns a dictionary containing four sample curves with the following
    structure:
        precision:
            1w:
                [(runtime,#states in intersection)...]
            2w:
                [...]
            3w:
                [...]
        recall:
            1w:
                [...]
            2w:
                [...]
            3w:
                [...]
    """
    randomer = Random(2222)

    # construct samples
    info("making samples...")
    samples = []
    step = int((len(log) + n_samples - 1) / n_samples)
    curr_size = step
    for i in range(n_samples):
        samples.append(sampler(log, curr_size, randomer))
        curr_size += step

    # check for samples dir
    safe_log_name = log.get_name().lower().replace(" ", "_")
    samples_dir = join(dump_directory, "samples", safe_log_name)
    if not exists(samples_dir):
        mkdir(samples_dir)

    # compute scores
    reporter.reset()
    percent_step = 100.0 / n_samples
    for i, sample in enumerate(samples):
        print(f"starting on sample {i} with size of {len(sample)}/{len(log)}...")

        # compute window versions
        for windows in range(1, 4, 1):
            print(f"starting windowing with {windows}...")
            reporter.shift_to_subpath(["recall", f"{windows}w", f"{(i+1)*percent_step:.0f}%"])

            for _ in range(4):
                _ = compute_stochastic_entropy_recall(
                    log,
                    sample,
                    window_size=windows,
                )

            reporter.shift_to_subpath(["precision", f"{windows}w", f"{(i+1)*percent_step:.0f}%"])

            for _ in range(4):
                _ = compute_stochastic_entropy_precision(
                    log,
                    sample,
                    window_size=windows,
                )
            print(f"completed windowing with {windows}...")

        print(f"completed sample ({i+1}/{len(samples)})...")

    return deepcopy(reporter._contexts)


def evaluation(log_paths: List[PathLike]):
    """
    Computes a sample curve on each of the given logs.
    """
    dump_directory = join(".", "evaluation", "shift-by-one")
    # clear_directory(dump_directory)

    for log_path in log_paths:
        print("working on log :: " + log_path + " ...")
        log = read_xes_simple(log_path)

        # swap out long activity names
        log, swaps = simplify_log_names(log)

        # read log and dump out stats
        info("collecting stats...")
        stats = {
            "name": log.get_name(),
            "size": len(log),
            "activities": repr(log.seen_activities()),
            "starts": repr(log.seen_start_activities()),
            "ends": repr(log.seen_end_activities()),
            "n_variants": log.get_nvariants(),
            "swaps": swaps,
        }
        with open(join(dump_directory, f"{log.get_name()}_stats.json"), "w") as f:
            f.write(dumps(stats, indent=4))

        # perform sampling
        info("performing sampling...")
        samples = 25
        runtimes_path = join(dump_directory, f"{log.get_name()}_s{samples}_runtimes.json")
        try:
            curves = sampling(log, samples, dump_directory=dump_directory)
        except KeyboardInterrupt as e:
            info("Cancellation detected, dumping partial reporter results...")
            curves = deepcopy(reporter._contexts)
            with open(runtimes_path, "w") as f:
                f.write(dumps(curves, indent=4))
            info("Partial runtime results dumped.")
            raise e
        
        with open(runtimes_path, "w") as f:
            f.write(dumps(curves, indent=4))

        info("Finished...")
        info(repr(curves))


if __name__ == "__main__":
    evaluation(EVAL_LOGS)
