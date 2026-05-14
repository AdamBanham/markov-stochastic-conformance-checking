from markov import (
    FiniteLabelledMarkovChain,
    convet_net_to_dot,
    convert_net_to_totally_defined,
    convert_net_to_be_recurrent,
    walk_and_assign_identifies
)

from pmkoalas.dtlog import convert


def create():
    """
    Tests what happens when we double totally defined after making a net
    recurrent. Uses a different langauge and log.
    """
    from os.path import join
    from util import clear_directory

    dump_directory = join(".", "sample_nets", "example_03b")
    clear_directory(dump_directory)

    traces = [""] + ["a f"] * 2 + ["a b"] * 6 + ["a b c"] * 6 + ["a c b"] * 7
    log = convert(*traces)

    example_net = FiniteLabelledMarkovChain()

    for trace, freq in log:
        example_net.add_variant(trace, freq)

    example_net = convert_net_to_totally_defined(example_net)
    example_net = convert_net_to_be_recurrent(example_net)
    example_net = convert_net_to_totally_defined(example_net)

    ids = walk_and_assign_identifies(example_net)
    # best display settings occurs with
    # min_len=4, mclimit=10000, rankdir=LR
    _ = convet_net_to_dot(example_net, 3, "LR", "example_three_b",
                          min_len=4, mclimit=10000,
                          identifiers=ids,
                          directory=dump_directory)


if __name__ == "__main__":
    create()
