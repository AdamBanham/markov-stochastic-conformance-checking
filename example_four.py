from markov import (
    convet_net_to_dot,
    convert_net_to_totally_defined,
    convert_net_to_be_recurrent,
    intersect_nets,
    walk_and_assign_identifies,
)
from discover import discover_chain_from_log
from measure import compute_stochastic_entropy_precision
from evaluation import sampler, simplify_log_names

from pmkoalas._logging import setLevel, info
from pmkoalas.dtlog import convert
from pmkoalas.read import read_xes_simple

setLevel("INFO")

from time import time
from os.path import join
from random import Random

LOG_FOLDER = join(".", "logs")


def create():
    """
    Tests the intersection of two recurrent and totally defined FLMCs.
    This example keeps the dead states and all states rather than all thoses
    states that are traversable from the initial state.
    """
    from os.path import join
    from util import clear_directory

    dump_directory = join(".", "sample_nets", "example_04")
    clear_directory(dump_directory)

    # left chain
    traces = [""] + ["z f"] * 2 + ["a b"] * 6 + ["a b c"] * 6 + ["a c b"] * 7
    a_log = convert(*traces)
    # a_log = read_xes_simple(join(LOG_FOLDER, "road_fines.xes"))
    # make naming safe
    # a_log,_ = simplify_log_names(a_log)
    a_example_net = discover_chain_from_log(a_log, window_length=3)

    # right chain
    traces = ["a b"] * 6 + ["a b c"] * 6 + ["a c b"] * 7
    b_log = convert(*traces)
    # construct sample
    # randomer = Random(2222)
    # step = int((len(a_log) + 20 - 1) / 20)
    # curr_size = step * 5
    # b_log = sampler(a_log, curr_size, randomer)

    b_example_net = discover_chain_from_log(b_log, window_length=3)

    # intersection prep
    shared_alphabet = a_example_net._alphabet.union(b_example_net._alphabet)
    a_example_net = convert_net_to_be_recurrent(a_example_net)
    b_example_net = convert_net_to_be_recurrent(b_example_net)

    # compute identifiers
    ids_a = walk_and_assign_identifies(a_example_net)
    ids_b = walk_and_assign_identifies(b_example_net)

    info("Example 4: making dot visualisations for a_net and b_net")
    convet_net_to_dot(
        a_example_net,
        3,
        "LR",
        "example_four_a",
        mclimit=10,
        min_len=2,
        identifiers=ids_a,
        directory=dump_directory,
    )
    convet_net_to_dot(
        b_example_net,
        3,
        "LR",
        "example_four_b",
        mclimit=10,
        min_len=2,
        identifiers=ids_b,
        directory=dump_directory,
    )

    # info("Example 4: performing runtime testing for intersecting...")
    # optimised_runs = []
    # not_optimised_runs = []

    # for _ in range(25):
    #     start = time()
    #     _ = intersect_nets(a_example_net, b_example_net)
    #     optimised_runs.append(time() - start)
    # mean_opt_time = sum(optimised_runs) / 25.0
    # info(f"mean optimised runtime :: {mean_opt_time}")

    # for _ in range(25):
    #     start = time()
    #     _ = intersect_nets(a_example_net, b_example_net, allow_optimisation=False)
    #     not_optimised_runs.append(time() - start)

    # mean_not_time = sum(not_optimised_runs) / 25.0

    # info(
    #     f"Optimised runtime improvements :: (opt) {mean_opt_time} vs  (not) {mean_not_time}"
    # )
    # info(f"Speed up :: {(mean_not_time/mean_opt_time)*100.0:.2f}%")

    info("Example: building intersection for vis...")
    ab_example_net = intersect_nets(a_example_net, b_example_net)
    ids_ab = walk_and_assign_identifies(ab_example_net)
    next_id = max(ids_ab.values()) + 1
    for state in ab_example_net._states:
        if state not in ids_ab:
            ids_ab[state] = next_id
            next_id += 1

    info("Example 4: making dot visualisations for ab_net")
    convet_net_to_dot(
        ab_example_net,
        3,
        "LR",
        "example_four_ab",
        mclimit=500,
        min_len=5,
        ranksep=1.2,
        size=1.2,
        node_fontsize=20,
        transition_fontsize=18,
        identifiers=ids_ab,
        directory=dump_directory,
    )

    precision = compute_stochastic_entropy_precision(a_log, b_log)
    info(f"precision computed :: {precision}")


if __name__ == "__main__":
    create()
