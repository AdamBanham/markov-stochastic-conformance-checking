from markov import (
    compute_long_run_proportions,
    walk_and_assign_identifies,
    convet_net_to_dot,
    intersect_nets
)
from mov_example_one import create_model_net
from mov_example_two import create_log_one_net, create_log_two_net
from discover import discover_chain_from_log
from measure import (
    compute_stochastic_entropy_precision,
    compute_stochastic_entropy_recall
)
from pmkoalas.dtlog import convert
from pmkoalas.simple import Trace
from pmkoalas._logging import setLevel
# setLevel('INFO')


def create():
    """Makes the intersections for the motivating example."""
    from os.path import join 
    from util import clear_directory

    dump_directory = join(".", "sample_nets", "motivated_example_03")
    # clear_directory(dump_directory)

    model = create_model_net()
    log_one = create_log_one_net()
    log_two = create_log_two_net()
    
    model_log_one = intersect_nets(model, log_one)
    model_log_two = intersect_nets(model, log_two)

    ids = walk_and_assign_identifies(model_log_one)
    _ = convet_net_to_dot(model_log_one, 3, "LR", "motivated_example_03_model_l1",
                          min_len=4, mclimit=10000,
                          ranksep=0.2,
                          size=0.65,
                          identifiers=ids,
                          directory=dump_directory)
    
    ids = walk_and_assign_identifies(model_log_two)
    _ = convet_net_to_dot(model_log_two, 3, "LR", "motivated_example_03_model_l2",
                          min_len=4, mclimit=10000,
                          size=0.65,
                          ranksep=0.2,
                          identifiers=ids,
                          directory=dump_directory)
    
    mapping, eqs = compute_long_run_proportions(model_log_one)
    print("model_log_1 long run probs :: ", mapping)

    mapping, eqs = compute_long_run_proportions(model_log_two)
    print("model_log_2 long run probs :: ", mapping)
    


if __name__ == "__main__":
    create()
