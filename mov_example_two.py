from markov import (
    compute_long_run_proportions,
    walk_and_assign_identifies,
    convet_net_to_dot
)
from mov_example_one import create_model_net
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
    """Tests making a FLMC from an event log using the prefix tree."""
    from os.path import join 
    from util import clear_directory

    dump_directory = join(".", "sample_nets", "motivated_example_02")
    # clear_directory(dump_directory)

    log = convert(*["a v c"])
    rlog= convert(*(["a v c"] + ["a v c f v c"]))

    source_log = create_model_net()
    left_net = discover_chain_from_log(log)
    print("left net long run :: ", compute_long_run_proportions(left_net))
    right_net = discover_chain_from_log(rlog) 
    right_long_runs, _ = compute_long_run_proportions(right_net)
    right_long_runs = dict(
        (k, round(float(v), 3))
        for k,v in right_long_runs.items()
    )
    print("right net long run :: ", right_long_runs)


    ids = walk_and_assign_identifies(left_net)
    _ = convet_net_to_dot(left_net, 3, "LR", "motivated_example_02_left",
                          min_len=4, mclimit=10000,
                          ranksep=0.2,
                          identifiers=ids,
                          directory=dump_directory)
    
    ids = walk_and_assign_identifies(right_net)
    _ = convet_net_to_dot(right_net, 3, "LR", "motivated_example_02_right",
                          min_len=4, mclimit=10000,
                          ranksep=0.2,
                          identifiers=ids,
                          directory=dump_directory)
    
    precision_left = compute_stochastic_entropy_precision(source_log, left_net)
    precision_right = compute_stochastic_entropy_precision(source_log, right_net)

    print(f"computed precision for L_1 :: {precision_left}")
    print(f"computed precision for L_2 :: {precision_right}")
    


if __name__ == "__main__":
    create()
