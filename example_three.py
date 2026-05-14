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
    recurrent.
    """
    from os.path import join 
    from util import clear_directory

    dump_directory = join(".", "sample_nets", "example_03")
    clear_directory(dump_directory)

    traces = ["a b"] * 6 + ["a b c"] * 6 + ["a c b"] * 7
    log = convert(*traces)

    example_net = FiniteLabelledMarkovChain()

    for trace, freq in log:
        example_net.add_variant(trace, freq)

    # best layouting occurred with digraph settings :
    # TB, minlen=5, mclimit=10000;
    example_net = convert_net_to_be_recurrent(example_net)

    ids = walk_and_assign_identifies(example_net)
    _ = convet_net_to_dot(example_net, 3, "LR", "example_three",
                          min_len=10, mclimit=100,
                          identifiers=ids,
                          directory=dump_directory)


if __name__ == "__main__":
    create()
