from markov import (
    FiniteLabelledMarkovChain,
    convet_net_to_dot,
    convert_net_to_totally_defined,
    convert_net_to_be_recurrent,
    make_one_step_transition_probabilties,
    make_latex_for_probabilties,
    intersect_nets,
)
from discover import discover_chain_from_log

from pmkoalas.dtlog import convert


def create():
    """
    Tests the intersection of two recurrent and totally defined FLMCs.
    This example reduces the intersection and computes the one-step
    probabilties.
    """
    from os.path import join
    from util import clear_directory

    dump_directory = join(".", "sample_nets", "example_05")
    clear_directory(dump_directory)

    # make left net
    traces = [""] + ["a f"] * 2 + ["a b"] * 6 + ["a b c"] * 6 + ["a c b"] * 7
    log = convert(*traces)
    example_net = discover_chain_from_log(log)
    example_net = convert_net_to_totally_defined(example_net)
    b_example_net = convert_net_to_be_recurrent(example_net)

    # make right net
    traces = ["a b"] * 6 + ["a b c"] * 6 + ["a c b"] * 7
    log = convert(*traces)
    example_net = discover_chain_from_log(log)
    example_net = convert_net_to_totally_defined(example_net)
    a_example_net = convert_net_to_be_recurrent(example_net)

    # make nets share an alphabet
    shared_alphabet = a_example_net._alphabet.union(b_example_net._alphabet)
    a_example_net = convert_net_to_totally_defined(a_example_net, shared_alphabet)
    b_example_net = convert_net_to_totally_defined(b_example_net, shared_alphabet)

    # make reduced intersection
    ab_example_net = intersect_nets(a_example_net, b_example_net)
    p_matrix, ids = make_one_step_transition_probabilties(ab_example_net)

    # create vis
    _ = convet_net_to_dot(
        ab_example_net,
        3,
        "LR",
        "example_five",
        mclimit=5000,
        min_len=5,
        size=0.8,
        node_fontsize=16,
        transition_fontsize=14,
        identifiers=ids,
        directory=dump_directory,
    )

    # make latex representation of proabilities
    latex_file = join(dump_directory, "example_five.latex")
    with open(latex_file, "w") as f:
        f.write(make_latex_for_probabilties(p_matrix, ids))


if __name__ == "__main__":
    create()
