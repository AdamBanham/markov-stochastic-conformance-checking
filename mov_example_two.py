from markov import (
    compute_long_run_proportions,
    walk_and_assign_identifies,
    convet_net_to_dot,
    FiniteLabelledMarkovChain
)
from mov_example_one import create_model_net
from discover import discover_chain_from_log
from measure import (
    compute_stochastic_entropy_precision,
)
from pmkoalas.dtlog import convert
from pmkoalas.simple import Trace
# setLevel('INFO')

def create_log_one_net() -> FiniteLabelledMarkovChain:
    log = convert(*["a v c"]) 
    net = discover_chain_from_log(log)

    state_mapping = {
        Trace(["a"]) : "2'",
        Trace(["a", "v"]) : "3'",
        Trace(["a", "v", "c"]) : "4'",
    }

    for state in net._states.difference(set([net._starting])):
        if state.trace in state_mapping:
            state.name = state_mapping[state.trace]

    net._starting.name = "1'"

    return net

def create_log_two_net():
    log = convert(*(["a v c"] + ["a v f v c"]))
    net =  discover_chain_from_log(log)

    suffix = "''"

    state_mapping = {
        Trace(["a"]) : "2",
        Trace(["a", "v"]) : "3",
        Trace(["a", "v", "c"]) : "4",
        Trace(["a", "v", "f"]) : "5",
        Trace(["a", "v", "f", "v"]) : "6",
        Trace(["a", "v", "f", "v", "c"]) : "7",
    }

    for state in net._states.difference(set([net._starting])):
        if state.trace in state_mapping:
            state.name = state_mapping[state.trace] + suffix

    net._starting.name = "1''"

    return net


def create():
    """Tests making a FLMC from an event log using the prefix tree."""
    from os.path import join 
    from util import clear_directory

    dump_directory = join(".", "sample_nets", "motivated_example_02")
    # clear_directory(dump_directory)

    # create log models and compute long run
    model = create_model_net()
    log_one = create_log_one_net()
    print("left net long run :: ", compute_long_run_proportions(log_one))

    log_two = create_log_two_net() 
    log_two_runs, _ = compute_long_run_proportions(log_two)
    log_two_runs = dict(
        (k, round(float(v), 3))
        for k,v in log_two_runs.items()
    )
    print("right net long run :: ", log_two_runs)

    # save out graphs
    ids = walk_and_assign_identifies(log_one)
    _ = convet_net_to_dot(log_one, 3, "LR", "motivated_example_02_left",
                          min_len=4, mclimit=10000,
                          ranksep=0.6,
                          transition_fontsize=16,
                          node_fontsize=16,
                          size=0.6,
                          identifiers=ids,
                          directory=dump_directory)
    
    ids = walk_and_assign_identifies(log_two)
    _ = convet_net_to_dot(log_two, 3, "LR", "motivated_example_02_right",
                          min_len=4, mclimit=10000,
                          transition_fontsize=16,
                          node_fontsize=16,
                          ranksep=0.6,
                          size=0.6,
                          identifiers=ids,
                          directory=dump_directory)
    
    precision_left = compute_stochastic_entropy_precision(log_one, model)
    precision_right = compute_stochastic_entropy_precision(log_two, model)

    print(f"computed precision for L_1 :: {precision_left}")
    print(f"computed precision for L_2 :: {precision_right}")
    

if __name__ == "__main__":
    create()
