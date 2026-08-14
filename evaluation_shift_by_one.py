from discover import discover_chain_from_log
from measure import (
    compute_stochastic_entropy_recall,
    compute_stochastic_entropy_precision,
)
from markov import (
    intersect_nets,
    convert_net_to_be_recurrent,
    convet_net_to_dot,
    convert_net_to_totally_defined,
)
from pmkoalas.read import read_xes_simple
from pmkoalas.export import export_to_xes_simple
from pmkoalas.simple import EventLog, Trace
from pmkoalas._logging import setLevel, info
from logging import INFO
from util import clear_directory

from os import PathLike, mkdir
from typing import List, Dict, Tuple
from os.path import join, exists
from random import Random
from json import dumps
from string import ascii_uppercase
from threading import Thread
from time import sleep
from itertools import product
from sys import set_int_max_str_digits

set_int_max_str_digits(18000)

LOG_FOLDER = join(".", "logs")
EVAL_LOGS = [
    join(LOG_FOLDER, "road_fines.xes"),
    join(LOG_FOLDER, "sepsis.xes"),
    join(LOG_FOLDER, "bpic_2020_permits.xes"),
]
# setLevel(INFO)
FULL_PREFIX = False
FACTOR = 10
FILTER = 0
DUMMMY_TRACE = Trace(["dummy"])

threads: List[Thread] = []


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

def resize_log(log: EventLog, factor:int=4) -> EventLog:
    """
    Resizes the log by some factor to ensure the slang is the same,
    but less prone to outliers or dropping of unique traces.
    """
    print("resize log with a factor of ", str(factor))
    traces = []
    for trace, freq in log:
        traces += [Trace(trace)] * (freq * factor)
    ret = EventLog(traces, log.get_name())
    print("finished resizing...")
    return ret

def drop_n_least_freq(log: EventLog, n:int=100):
    """
    Reshapes the log to not include the n-least freq variants in the log. 
    """
    print(f"filtering the {n} least freq variants from log...")
    traces = sorted(log._freqset.items(), reverse=True, key=lambda x: x[1])
    traces = traces[:-n]
    traces = [
        trace
        for trace, freq
        in traces
        for _ in range(freq)
    ]
    ret = EventLog(traces, log.get_name())
    print(f"Log had {log.get_nvariants()} but now contains {ret.get_nvariants()}")
    return ret



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
    dummy = DUMMMY_TRACE
    pool.append((dummy, 0))

    # pick traces
    for _ in range(size):
        pick = int(randomer.choice(list(pool_size)))
        trace, count = pool[pick]
        if count > 0:
            pool[pick] = (trace, max(0, count - 1))
            _, count = pool[-1]
            pool[-1] = (dummy, count+1)
        else:
            print(f"picked a trace that did not have a non-zero count :: {trace, count}")

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


def sampling(log: EventLog, n_samples: int, dump_directory: PathLike, full_prefix:bool=False) -> Dict:
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
                [...]
            3w:
                [...]
            5w:
                [...]
            7w:
                [...]
        recall:
            1w:
                [...]
            3w:
                [...]
            5w:
                [...]
            7w:
                [...]
    """
    randomer = Random(2222)

    # construct samples
    print("making samples...")
    samples: List[EventLog] = []
    step = int((len(log) + n_samples - 1) / n_samples)
    curr_size = 0
    for i in range(n_samples+1):
        samples.append(sampler(log, curr_size, randomer))
        curr_size += step

    # ret dictionaries
    precision = {
        "1w": [],
        "2w": [],
        "3w": [],
        "5w": [],
        "7w": [],
        "full" : [],
    }
    recall = {
        "1w": [],
        "2w": [],
        "3w": [],
        "5w": [],
        "7w": [],
        "full" : []
    }

    # check for samples dir
    safe_log_name = log.get_name().lower().replace(" ", "_")
    samples_dir = join(dump_directory, "samples", safe_log_name)
    if not exists(samples_dir):
        mkdir(samples_dir)

    # compute scores
    print("starting measurements...")
    try:
        for i, sample in enumerate(samples):

            sample_size = sample._freqset[DUMMMY_TRACE] if DUMMMY_TRACE in sample._freqset else 0
            print(f"starting on sample {i} with number of dummy traces as {sample_size}/{len(log)}...")

            # export_to_xes_simple(
            #     join(samples_dir, f"s{i:02d}_event_log.xes"),
            #     sample
            # )
            # with open(join(samples_dir, f"s{i:02d}_event_log.eval"), "w") as f:
            #     f.write(repr(sample))

            if full_prefix:
                print("computing measures using full_prefix (this may take some time)...")
                print("computing recall...")
                recall_score, thread = compute_stochastic_entropy_recall(
                    log, sample,
                ) 
                print("computed recall...")
                
                print("computing precision...")
                precision_score, thread = compute_stochastic_entropy_precision(
                    log, sample,
                )
                print("computed precision...")
                recall["full"].append(recall_score)
                precision["full"].append(precision_score)
            else:
                # compute window versions
                for windows in range(1, 4, 2):
                    print(f"starting windowing with {windows}...")

                    print("computing recall...")
                    recall_score, thread = compute_stochastic_entropy_recall(
                        log, sample, window_size=windows,
                        # dump_location=samples_dir,
                        # dump_filename=f"s{i:02d}_w{windows}_recall"
                    )
                    # threads.append(thread)
                    print("computed recall...")

                    print("computing precision...")
                    precision_score, thread = compute_stochastic_entropy_precision(
                        log, sample, window_size=windows,
                        # dump_location=samples_dir,
                        # dump_filename=f"s{i:02d}_w{windows}_precision"
                    )
                    # threads.append(thread)
                    print("computed precision...")

                    recall[f"{windows}w"].append(recall_score)
                    precision[f"{windows}w"].append(precision_score)
                    print(f"completed windowing with {windows}...")

            print(f"completed sample ({i+1}/{len(samples)})...")
    except:
        print("breaking and returning...")
        pass

    return {"precision": precision, "recall": recall}


def evaluation(log_paths: List[PathLike], full_prefix_length:bool=False):
    """
    Computes a sample curve on each of the given logs.
    """
    dump_directory = join(".", "evaluation", "shift-by-one")
    # clear_directory(dump_directory)

    for log_path in log_paths:
        info("working on log :: " + log_path + " ...")
        log = read_xes_simple(log_path)

        # swap out long activity names
        if FILTER > 0:
            log = drop_n_least_freq(log, FILTER)
        log, swaps = simplify_log_names(log)
        log = resize_log(log, factor=FACTOR)

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
        with open(join(dump_directory, f"{log.get_name()}_stats_v4.json"), "w") as f:
            f.write(dumps(stats, indent=4))

        # perform sampling
        info("performing sampling...")
        curves = sampling(log, 25, dump_directory=dump_directory, full_prefix=full_prefix_length)
        with open(join(dump_directory, f"{log.get_name()}_scores_v4.json"), "w") as f:
            f.write(dumps(curves, indent=4))

        info("Finished...")
        info(repr(curves))

    info("Finished computation, waiting for dot to graph...")
    # wait for dot to finish
    while True:
        finished = all([not t.is_alive() for t in threads])
        if finished:
            break
        sleep(15.0)
        info("still waiting...")


if __name__ == "__main__":
    evaluation(EVAL_LOGS, FULL_PREFIX)
