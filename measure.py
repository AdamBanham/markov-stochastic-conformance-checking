from markov import (
    FiniteLabelledMarkovChain,
    intersect_nets,
    compute_long_run_proportions,
    convert_net_to_totally_defined,
    convert_net_to_be_recurrent,
    make_latex_for_probabilties,
    convet_net_to_dot,
    walk_and_assign_identifies
)
from discover import discover_chain_from_log
from pmkoalas.simple import EventLog
from pmkoalas._logging import info
from copy import deepcopy
from json import dumps
from threading import Thread
from os import PathLike
from typing import Tuple, Union


def compute_stochastic_entropy_recall(
    left: EventLog | FiniteLabelledMarkovChain,
    right: EventLog | FiniteLabelledMarkovChain,
    window_size=-1,
    dump_location: str | PathLike=None,
    dump_filename: str=None
) -> Tuple[float, Union[None, Thread]]:
    """
    Computes stochastic entropy recall on the given inputs.
    """
    dump_thread = None
    # discover nets
    info("starting processing of logs...")
    if isinstance(left, EventLog):
        left_net = discover_chain_from_log(left, window_length=window_size)
    else:
        left_net = left
    
    if isinstance(right, EventLog):
        right_net = discover_chain_from_log(right, window_length=window_size)
    else:
        right_net = right
    info("finished processing of logs...")

    info("starting processing for intersection...")
    # # make both totally defined on the shared alphabet
    # shared_alphabet = left_net._alphabet.union(right_net._alphabet)
    # t_left_net = convert_net_to_totally_defined(left_net, shared_alphabet)
    # t_right_net = convert_net_to_totally_defined(right_net, shared_alphabet)

    # make them both recurrent
    tr_left_net = convert_net_to_be_recurrent(left_net)
    tr_right_net = convert_net_to_be_recurrent(right_net)
    info("finished processing for intersection...")

    # compute probs for left
    info("starting compute to left long run proportions...")
    left_probs, eqs = compute_long_run_proportions(tr_left_net)
    info("A :: solved equations :: " + repr(eqs))
    info("A :: long run proportions computed :: " + repr(left_probs))
    info("finished compute to left long run proportions...")

    # compute distance between chains
    info("starting intersection...")
    ab_example_net = intersect_nets(tr_left_net, tr_right_net)
    if dump_location:
        dump_thread = Thread(
            target=convet_net_to_dot, 
            kwargs={
                "net": ab_example_net, 
                "directory": dump_location,
                "file_name": dump_filename if dump_filename else "b_x_a_intersection",
                "size": 0.8
            }
        )
        dump_thread.start()
    info("finished intersection...")

    # edge case if the intersection is without any outgoing then return 0
    if len(ab_example_net.actions_from(ab_example_net._starting)) < 1:
        info("edge case occured, the intersection had no edges.")
        info(f"recall was computed as :: {0.0:.6f}")
        return 0.0, dump_thread

    info("starting compute to long run proportions for intersection...")
    ab_probs, eqs = compute_long_run_proportions(ab_example_net)
    info("A X B :: solved equations :: " + repr(eqs))
    info("A X B :: long run proportions computed :: " + repr(ab_probs))
    info("finished compute to long run proportions for intersection...")

    info("Stochastic entropy recall using intersection of A X B...")
    recall = 1.000
    delta = 0.0
    left_factor = 1 / (1 - left_probs[tr_left_net._starting])
    right_factor = 1 / (1 - ab_probs[ab_example_net._starting])
    
    for state in tr_left_net._states:
        if state == tr_left_net._starting:
            continue

        state_prob = left_factor * left_probs[state]
        info(f"processing left state :: {state} with probability :: {state_prob:.6f}")

        intersect_prob = 0.0
        for other in ab_example_net._states:
            if other.left != state:
                continue
            if other.right == tr_right_net._starting:
                continue
            
            info(f"processing intersection state :: {other} with probability :: {ab_probs[other]:.6f}")
            intersect_prob += ab_probs[other]
        intersect_prob = right_factor * intersect_prob

        info(f"total intersect probability for left state :: {state} is :: {intersect_prob:.6f}")
        step = state_prob - intersect_prob
        step = max(step, 0)
        delta += step
        info(f"delta is :: {delta:.6f}")

    recall = recall - delta
    info(f"recall was computed as :: {recall:.6f}")
    return float(recall), dump_thread


def compute_stochastic_entropy_precision(
    left: EventLog | FiniteLabelledMarkovChain,
    right: EventLog | FiniteLabelledMarkovChain,
    window_size=-1,
    dump_location: str | PathLike=None,
    dump_filename: str=None
) -> Tuple[float, Union[None, Thread]]:
    """
    Computes stochastic entropy recall on the given inputs.
    """
    dump_thread = None 
    # discover nets
    info("starting processing of logs...")
    if isinstance(left, EventLog):
        left_net = discover_chain_from_log(left, window_length=window_size)
    else:
        left_net = left

    if isinstance(right, EventLog):
        right_net = discover_chain_from_log(right, window_length=window_size)
    else:
        right_net = deepcopy(right)
    info("finished processing of logs...")

    info("starting processing for intersection...")
    # # make both totally defined on the shared alphabet
    # shared_alphabet = left_net._alphabet.union(right_net._alphabet)
    # t_left_net = convert_net_to_totally_defined(left_net, shared_alphabet)
    # t_right_net = convert_net_to_totally_defined(right_net, shared_alphabet)

    # make them both recurrent
    tr_left_net = convert_net_to_be_recurrent(left_net)
    tr_right_net = convert_net_to_be_recurrent(right_net)

    info("finished processing for intersection...")

    # compute probs for left
    info("starting compute to left long run proportions...")
    right_probs, eqs = compute_long_run_proportions(tr_right_net)
    info("B :: solved equations :: " + repr(eqs))
    info("B :: long run proportions computed :: " + repr(right_probs))
    info("finished compute to left long run proportions...")

    # compute distance between chains
    info("starting intersection...")
    ba_example_net = intersect_nets(tr_right_net, tr_left_net)
    if dump_location:
        dump_thread = Thread(
            target=convet_net_to_dot, 
            kwargs={
                "net": ba_example_net, 
                "directory": dump_location,
                "file_name": dump_filename if dump_filename else "b_x_a_intersection",
                "size": 0.8
            }
        )
        dump_thread.start()
    info("finished intersection...")

    # edge case if the intersection is without any outgoing then return 0
    if len(ba_example_net.actions_from(ba_example_net._starting)) < 1:
        info("edge case occured, the intersection had no edges.")
        info(f"recall was computed as :: {0.0:.6f}")
        return 0.0, dump_thread

    info("starting compute to long run proportions for intersection...")
    ba_probs, eqs = compute_long_run_proportions(ba_example_net)
    info("B X A :: solved equations :: " + repr(eqs))
    info("B X A :: long run proportions computed :: " + repr(ba_probs))
    info("finished compute to long run proportions for intersection...")
    
    info("Stochastic entropy precision using intersection of B X A...")
    precision = 1.00
    delta = 0.0
    left_factor = 1 / (1 - right_probs[tr_right_net._starting])
    right_factor = 1 / (1 - ba_probs[ba_example_net._starting])

    # work out the sum
    for state in tr_right_net._states:
        if state == tr_right_net._starting:
            continue

        state_prob = left_factor * right_probs[state]
        info(f"processing left state :: {state} with probability :: {state_prob:.3f}")
        
        intersect_prob = 0.0
        for other in ba_example_net._states:
            if other.left != state:
                continue
            if other.right == tr_left_net._starting:
                continue
            info(f"processing intersection state :: {other} with probability :: {ba_probs[other]:.3f}")
            intersect_prob += ba_probs[other]
        intersect_prob = right_factor * intersect_prob
        
        info(f"total intersect probability for left state :: {state} is :: {intersect_prob:.3f}")
        step = state_prob - intersect_prob
        step = max(step, 0)
        delta += step
        info(f"delta is :: {delta:.3f}")

    precision = precision - delta
    info(f"precision was computed as :: {precision:.3f}")
    return float(precision), dump_thread
