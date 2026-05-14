from markov import (
    FiniteLabelledMarkovChain,
    convet_net_to_dot,
    walk_and_assign_identifies
)

from pmkoalas.dtlog import convert


def create():
    """Tests making a FLMC from an event log using the prefix tree."""
    from os.path import join 
    from util import clear_directory

    dump_directory = join(".", "sample_nets", "example_01")
    clear_directory(dump_directory)

    traces = ["a b"] * 6 + ["a b c"] * 6 + ["a c b"] * 7
    log = convert(*traces)

    example_net = FiniteLabelledMarkovChain()

    for trace, freq in log:
        example_net.add_variant(trace, freq)

    ids = walk_and_assign_identifies(example_net)
    _ = convet_net_to_dot(example_net, 3, "LR", "example_one",
                          min_len=4, mclimit=100,
                          identifiers=ids,
                          directory=dump_directory)
    print(ids)

if __name__ == "__main__":
    create()
