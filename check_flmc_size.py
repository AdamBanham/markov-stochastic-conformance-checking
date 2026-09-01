from os.path import join
from json import dumps
from pmkoalas.read import read_xes_simple
from discover import discover_chain_from_log
from markov import convet_net_to_dot

from evaluation_shift_by_one import simplify_log_names

LOG_FOLDER = join(".", "logs")
EVAL_LOGS = [
    join(LOG_FOLDER, "road_fines.xes"),
    join(LOG_FOLDER, "sepsis.xes"),
    join(LOG_FOLDER, "bpic_2020_permits.xes"),
]
SOCIAL_LOGS = [
    join(LOG_FOLDER, "brazil_1.xes"),
    join(LOG_FOLDER, "honduras_coordinated.xes"),
    join(LOG_FOLDER, "uae_coordinated.xes"),
]

LOGS = EVAL_LOGS + SOCIAL_LOGS


def work():
    full_results = {}
    w1_results = {}
    w2_results = {}
    w3_results = {}

    for log_path in LOGS:
        print("working on log :: " + log_path + " ...")
        log = read_xes_simple(log_path)
        simple_log, _ = simplify_log_names(log)

        net = discover_chain_from_log(simple_log)

        print(f"{log.name} has a size of {len(net._states)}")

        full_results[log.name] = len(net._states)

        # convet_net_to_dot(net, file_name=join(".", f"{log.name}.dot"))

        for results, window in zip(
            [w1_results, w2_results, w3_results],
            [1, 2, 3]
        ):
            net = discover_chain_from_log(simple_log, window_length=window)            
            print(f"(window={window}) {log.name} has a size of {len(net._states)}")
            results[log.name] = len(net._states)
            # convet_net_to_dot(net, file_name=join(".", f"{log.name}.w{window}.dot"))

    with open(join(".", "flmc_sizes.json"), "w") as f:
        f.write(dumps(full_results, indent=4))

    for results, window in zip(
        [w1_results, w2_results, w3_results],
        [1, 2, 3]
    ):
        with open(join(".", f"flmc_sizes_window_{window}.json"), "w") as f:
            f.write(dumps(results, indent=4))


if __name__ == "__main__":
    work()
