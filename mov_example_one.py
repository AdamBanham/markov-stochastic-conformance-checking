from markov import (
    FiniteLabelledMarkovChain,
    MarkovState,
    convet_net_to_dot,
    walk_and_assign_identifies,
    STOP_SYMBOL,
    compute_long_run_proportions
)
from discover import discover_chain_from_log

from pmkoalas.simple import Trace

def create_model_net() -> FiniteLabelledMarkovChain:
    example_net = FiniteLabelledMarkovChain()
    state_a =  MarkovState("1")
    state_v = MarkovState("2")
    state_c = MarkovState("3")
    state_f = MarkovState("4")

    example_net._alphabet.add("a")
    example_net._alphabet.add("v")
    example_net._alphabet.add("c")
    example_net._alphabet.add("f")

    example_net._states.add(
       state_a
    )
    example_net._states.add(
       state_v
    )
    example_net._states.add(
       state_c
    )
    # example_net._states.add(
    #     state_f
    # )
    
    example_net._accepting.add(state_c)
    
    example_net._transitions[example_net._starting]['a'] = state_a
    example_net._transitions[state_a] = {
        'v' : state_v
    }
    example_net._transitions[state_v] = {
        'c' : state_c, 'f': state_a
    }
    # example_net._transitions[state_f] = {
    #     'a': state_a
    # }
    example_net._transitions[state_c] = {
        STOP_SYMBOL : example_net._starting,
        
    }

    example_net._counting[example_net._starting]['a'] = 1
    example_net._counting[state_a] = {
        'v' : 1
    }
    example_net._counting[state_v] = {
        'c' : 1, 'f': 1
    }
    example_net._counting[state_f] = {
        'a' : 1
    }
    example_net._counting[state_c] = {
        STOP_SYMBOL: 1
    }
    return example_net

def create():
    """Tests making a FLMC from an event log using the prefix tree."""
    from os.path import join 
    from util import clear_directory

    dump_directory = join(".", "sample_nets", "motivated_example_01")
    # clear_directory(dump_directory)

    example_net = create_model_net()

    ids = walk_and_assign_identifies(example_net)
    _ = convet_net_to_dot(example_net, 3, "LR", "motivated_example_01",
                          min_len=4, mclimit=10000,
                          ranksep=0.2,
                          identifiers=ids,
                          directory=dump_directory)
    
    print(compute_long_run_proportions(example_net))

if __name__ == "__main__":
    create()
