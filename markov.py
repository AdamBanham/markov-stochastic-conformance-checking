from dataclasses import dataclass
from typing import Set, Dict, Tuple, List, Union, TypeVar
from pmkoalas.simple import Trace, EventLog
from pmkoalas._logging import info
from graphviz import Digraph
from copy import deepcopy
from collections import deque

import numpy as np

from sympy import Symbol, Eq, solve, Rational, linsolve


def proc(label: str) -> str:
    return label.replace("<", "&lt;").replace(">", "&gt;")


@dataclass
class MarkovState:
    trace: Trace

    def __hash__(self) -> int:
        return hash(self.name)

    @property
    def name(self):
        return str(self.trace)

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, value: object) -> bool:
        if isinstance(value, MarkovState):
            return self.trace == value.trace
        return False

    def dot_label(self, top_bot=True) -> str:
        return proc(str(self.trace))

    def dot_shape(self) -> str:
        return "circle"


@dataclass
class IntersectedMarkovState:
    left: MarkovState
    right: MarkovState

    def __hash__(self) -> int:
        return hash(self.name)

    @property
    def name(self):
        if self.left.name != self.right.name:
            return f"L={self.left},R={self.right}."
        else:
            return f"(L&amp;R){self.left}"

    def __repr__(self) -> str:
        return self.name

    def dot_label(self, top_bot=True) -> str:
        if self.left != self.right:
            if top_bot:
                return "{" + f"{proc(str(self.left))}|{proc(str(self.right))}" + "}"
            else:
                return (
                    "{"
                    + f"{proc(str(self.left))}"
                    + "}|"
                    + "{"
                    + f"{proc(str(self.right))}"
                    + "}"
                )
        else:
            return "{" + f"{proc(str(self.left))}" + "}"

    def dot_shape(self) -> str:
        return "Mrecord"


EMPTY_TRACE_SYMBOL = "&epsilon;"
EMPTY_TRACE_STATE = MarkovState(EMPTY_TRACE_SYMBOL)

DEFAULT_START_STATE = EMPTY_TRACE_STATE

DEAD_STATE_SYMBOL = "&omega;"
DEAD_STATE = MarkovState(DEAD_STATE_SYMBOL)

STOP_SYMBOL = "&chi;"
SILENT_SYMBOL = "&tau;"

NOT_SANS_SYMBOLS = [STOP_SYMBOL, SILENT_SYMBOL, EMPTY_TRACE_SYMBOL, DEAD_STATE_SYMBOL]


class FiniteLabelledMarkovChain:
    """
    This abstractions allows for the construction of a markov chain by passing
    known variants of a log.
    """

    def __init__(self):
        self._starting: MarkovState = DEFAULT_START_STATE
        self._states: Set[MarkovState] = set([self._starting])
        self._accepting: Set[MarkovState] = set()
        self._alphabet: Set[str] = set(
            [
                SILENT_SYMBOL,
                STOP_SYMBOL,
            ]
        )

        self._seen_sentences = set()

        self._counting: Dict[MarkovState, Dict[str, int]] = {self._starting: {}}
        self._transitions: Dict[MarkovState, Dict[str, MarkovState]] = {
            self._starting: {}
        }

    def probability(self, state: MarkovState, word: str) -> float:
        """
        Computes the probability of transition from the given state, with
        the given word.
        """
        if state not in self._states:
            raise ValueError(f"The given state is not known to this net :: {state}")

        options = self._counting[state]
        total_weight = sum(options.values())
        word_weight = options[word] if word in options else 0

        return word_weight / total_weight if total_weight > 0 else 0

    def moves_to(self, state: MarkovState, word: str) -> MarkovState:
        """
        Returns the next state from the given state when transitioning using
        the given word.
        """
        if state not in self._states:
            raise ValueError(f"The given state is not known to this net :: {state}")

        moves = self._transitions[state]

        if word not in moves:
            raise ValueError(
                "The given word cannot be transitioned from the state"
                + f":: {state} to {word} (options={repr(moves)})"
            )

        return moves[word]

    def actions_from(self, state: MarkovState) -> Set[str]:
        """
        Returns the set of outgoing words from this state.
        """
        if state not in self._states:
            raise ValueError(
                f"The given state is not known to the net :: {repr(state)}"
            )

        ret = set()
        for word in self._transitions[state]:
            ret.add(word)
        return ret

    def likelihood_of(self, variant: Trace) -> float:
        """
        Computes the likelihood of seeing this variant.
        Unknown variants are defaulted to zero.
        """
        prob = 1.0

        curr = self._starting
        for action in variant:
            pass

        return prob

    def add_variant(self, variant: Trace, freq: int, window_length=-1):
        """
        Expands the net to include transitions and states to account for the
        given variant.
        Updates probability based on the given observation count.
        """
        # update actions
        self._alphabet = self._alphabet.union(variant._acts)

        # edge case: handle empty traces
        # maybe not needed until we get to recurrent?
        if len(variant) == 0:
            self._alphabet.add(STOP_SYMBOL)
            self._accepting.add(self._starting)

            moves = self._transitions[self._starting]
            counts = self._counting[self._starting]

            if STOP_SYMBOL not in moves:
                moves[STOP_SYMBOL] = self._starting
                counts[STOP_SYMBOL] = freq
            else:
                counts[STOP_SYMBOL] += freq
            curr = self._starting
        else:
            # walk and add states and transitions as needed
            curr = self._starting
            walking_trace = deque([], maxlen=window_length) if window_length > 0 else []
            for act in variant:

                walking_trace.append(act)

                try:
                    next = self.moves_to(curr, act)
                except ValueError:
                    # create transition
                    next = MarkovState(Trace([act for act in walking_trace]))

                    # update net
                    self._states.add(next)
                    moves = self._transitions[curr]
                    moves[act] = next
                    if next not in self._transitions:
                        self._transitions[next] = {}
                    if next not in self._counting:
                        self._counting[next] = {}

                # update the count
                counts = self._counting[curr]
                if act not in counts:
                    counts[act] = freq
                else:
                    counts[act] += freq

                # move to next
                curr = next

        # set the last state as accepting
        self._accepting = self._accepting.union(set([curr]))

        # add stop symbol for the last step
        self._alphabet.add(STOP_SYMBOL)
        if curr in self._states:
            moves = self._transitions[curr]
            counts = self._counting[curr]

            if STOP_SYMBOL not in moves:
                moves[STOP_SYMBOL] = self._starting
                counts[STOP_SYMBOL] = freq
            elif STOP_SYMBOL in moves and moves[STOP_SYMBOL] == self._starting:
                counts[STOP_SYMBOL] += freq
            elif self.probability(curr, STOP_SYMBOL) == 0:
                moves[STOP_SYMBOL] = self._starting
                counts[STOP_SYMBOL] = freq

        self._seen_sentences.add(variant)

    def __str__(self) -> str:
        ret = "FLMC:=\n"
        ret += f"\tS:={repr(self._states)}\n"
        ret += f"\ts_0:={repr(self._starting)}\n"
        ret += f"\tF:={repr(self._accepting)}\n"
        ret += f"\tA:={repr(self._alphabet)}\n"

        ret += "\tprob:=\n"
        for state in self._counting:
            ret += f"\t\t{repr(state)}:="

            if len(self._counting[state].values()) < 1:
                ret += "{},\n"
                continue

            ret += "{"
            total_weight = sum(self._counting[state].values())
            for word, wieght in self._counting[state].items():
                ret += f"{wieght/total_weight:.2f}->{word}, "

            ret = ret[:-2]
            ret += "}\n"

        ret += f"\ttrans:={repr(self._transitions)}\n"
        return ret


class IntersectedFiniteLabelledMarkovChain(FiniteLabelledMarkovChain):
    """
    This class represents the intersection between two totally defined FLMCs.
    """

    def __init__(
        self,
        left_net: FiniteLabelledMarkovChain,
        right_net: FiniteLabelledMarkovChain,
        keep_traversable: bool = True,
        keep_dead_states: bool = False,
        allow_optimisation: bool = True
    ):
        super().__init__()

        self._left_net = deepcopy(left_net)
        self._right_net = deepcopy(right_net)

        self._starting: IntersectedMarkovState = IntersectedMarkovState(
            left_net._starting, right_net._starting
        )
        self._states: Set[IntersectedMarkovState] = set([self._starting])
        self._accepting: Set[IntersectedMarkovState] = set()
        self._alphabet = left_net._alphabet.union(right_net._alphabet)
        self._counting: Dict[IntersectedMarkovState, Dict[str, int]] = {
            self._starting: {}
        }
        self._transitions: Dict[
            IntersectedMarkovState, Dict[str, IntersectedMarkovState]
        ] = {self._starting: {}}

        # if the defaults are used, we can optimise the construction.
        if keep_traversable and not keep_dead_states and allow_optimisation:
            info("computing optimised walk intersection...")
            self._optimised_walk_construction()
            return

        # compute states
        for left_state in left_net._states:
            for right_state in right_net._states:
                if not keep_dead_states and (
                    left_state == DEAD_STATE or right_state == DEAD_STATE
                ):
                    continue
                new_state = IntersectedMarkovState(left_state, right_state)
                self._states.add(new_state)

                if new_state in self._transitions:
                    continue

                self._transitions[new_state] = {}
                self._counting[new_state] = {}

                # check for accepting
                if (
                    new_state.left in left_net._accepting
                    and new_state.right in right_net._accepting
                ):
                    self._accepting.add(new_state)

        # compute transitions
        info(f"computing transitions for {len(self._states)=}")
        for state in self._states:
            moves = self._transitions[state]
            for letter in self._alphabet:
                try:
                    left_state = self._left_net.moves_to(state.left, letter)
                except ValueError:
                    continue
                try:
                    right_state =  self._right_net.moves_to(state.right, letter)
                except ValueError:
                    continue
                new_state = IntersectedMarkovState(
                    left_state,
                    right_state,
                )

                if new_state in self._states:
                    moves[letter] = new_state

        # find the traversed states from the starting,
        # remove all non traversed states
        removed = set()
        if keep_traversable:
            traversed = set()
            queue = [self._starting]

            while len(queue) > 0:
                info(
                    f"intersection que size: {len(queue)} but has seen {len(traversed)=}"
                )
                curr = queue.pop(0)

                if curr in traversed:
                    continue

                traversed.add(curr)

                if curr not in self._transitions:
                    print(f"could not find transitions for {curr=}")
                    continue

                for state in self._transitions[curr].values():
                    if state not in traversed:
                        queue.append(state)

            # update states to only keep traversed
            removed = self._states.difference(traversed)
            self._states = traversed
            self._accepting = self._accepting.intersection(traversed)
            # update transitions so we don't go to these states.
            for remove in removed:
                self._transitions.pop(remove)
                self._counting.pop(remove)
            # update remaining transitions
            for state in self._states:
                transitions = self._transitions[state]
                flipped = dict((value, key) for key, value in transitions.items())
                for remove in removed:
                    if remove in flipped:
                        t_key = flipped[remove]
                        if t_key in transitions:
                            transitions.pop(t_key)

        # for all states that do not sum to one, remove them
        removed = set()
        for state in self._states:
            sum = 0.0
            for letter in self._alphabet:
                sum += self.probability(state, letter)
            if sum < 0.99:
                print(f"removing {state} because {sum=}")
                removed.add(state)
        # remove state
        self._states.difference_update(removed)
        self._accepting.difference_update(removed)
        # update transitions so we don't go to these states.
        for remove in removed:
            if remove in self._transitions:
                self._transitions.pop(remove)
        # update remaining transitions
        for state in self._states:
            transitions = self._transitions[state]
            flipped = dict((value, key) for key, value in transitions.items())
            for remove in removed:
                if remove in flipped:
                    t_key = flipped[remove]
                    if t_key in transitions:
                        transitions.pop(t_key)

    def _optimised_walk_construction(self):
        """
        Breadth depth search from the initial state to construct a
        recurrent and reduciable intersection.
        """

        traversed = set()
        queue = [self._starting]

        while len(queue) > 0:
            info(f"intersection que size: {len(queue)} but has seen {len(traversed)=}")
            curr = queue.pop(0)

            if curr in traversed:
                continue

            traversed.add(curr)

            # spawn next states
            spawn_moves = {}
            accepts = set()
            spawned = []
            saw_non_zero = False
            for letter in self._alphabet:
                try:
                    left_state = self._left_net.moves_to(curr.left, letter)
                except ValueError:
                    continue
                try:
                    right_state = self._right_net.moves_to(curr.right, letter)
                except ValueError:
                    continue

                # check for dead states
                if left_state == DEAD_STATE or right_state == DEAD_STATE:
                    continue

                # make state
                new_state = IntersectedMarkovState(left_state, right_state)
                spawned.append(new_state)

                # check for accepting
                if (
                    new_state.left in self._left_net._accepting
                    and new_state.right in self._right_net._accepting
                ):
                    accepts.add(new_state)

                # update moves
                spawn_moves[letter] = new_state

                # check probability
                word_weight = min(
                    self._left_net.probability(curr.left, letter),
                    self._right_net.probability(curr.right, letter),
                )
                saw_non_zero = saw_non_zero or word_weight > 0

            # if we saw probability
            if saw_non_zero:
                self._states.add(curr)
                self._accepting = self._accepting.union(accepts)
                self._transitions[curr] = spawn_moves
                self._counting[curr] = {}
                queue += spawned
        
    def probability(self, state: IntersectedMarkovState, word: str) -> float:
        """
        Computes the probability of transition from the given state, with
        the given word.
        """
        if state not in self._states:
            raise ValueError(f"The given state is not known to this net :: {state}")

        actions = self.actions_from(state)
        total_weight = sum(
            min(
                self._left_net.probability(state.left, letter),
                self._right_net.probability(state.right, letter),
            )
            for letter in actions
        )
        word_weight = min(
            self._left_net.probability(state.left, word),
            self._right_net.probability(state.right, word),
        )

        return word_weight / total_weight if total_weight > 0 else 0


NetTypes = TypeVar(
    "NetTypes", FiniteLabelledMarkovChain, IntersectedFiniteLabelledMarkovChain
)


def convert_net_to_totally_defined(
    net: NetTypes,
    alphabet: Set | None = None,
) -> NetTypes:
    """
    Adds a dead state to the net and for all states which do not transition
    to a letter in the alphabet of the net, adds a transition and probablity
    from that state to the new dead state.
    """
    ret = deepcopy(net)

    if alphabet is None:
        alphabet = ret._alphabet

    add_dead_state = False
    # now for each state, make a connection to the dead state as needed
    for state in ret._states:
        if state == DEAD_STATE_SYMBOL:
            continue

        actions = ret.actions_from(state)
        missing = alphabet.difference(actions)
        transitions = ret._transitions[state]
        counts = ret._counting[state]

        for missing_action in missing:
            transitions[missing_action] = DEAD_STATE
            counts[DEAD_STATE_SYMBOL] = 0

            add_dead_state = True

    if add_dead_state:
        # add dead state
        ret._states.add(DEAD_STATE)

        # check for transitions
        if DEAD_STATE not in ret._transitions:
            ret._transitions[DEAD_STATE] = dict(
                (letter, DEAD_STATE) for letter in alphabet
            )
        else:
            for letter in ret._alphabet:
                if letter not in ret._transitions[DEAD_STATE]:
                    ret._transitions[DEAD_STATE][letter] = DEAD_STATE

        # check for counting
        if DEAD_STATE not in ret._counting:
            ret._counting[DEAD_STATE] = dict((letter, 0) for letter in alphabet)
        else:
            for letter in ret._alphabet:
                if letter not in ret._counting[DEAD_STATE]:
                    ret._counting[DEAD_STATE][letter] = 0

    return ret


def convert_net_to_be_recurrent(net: NetTypes) -> NetTypes:
    """
    Converts a net to be recurrent and makes use of the frequencies of
    variants to include stopping transitions. Otherwise, silent transitions
    are used to complete the recurrent transformation.

    Returns a new instance of the net that is recurrent.
    """
    ret = deepcopy(net)
    # check for states that match the variants observed in the log
    # adding stopping transitions back to the starting state.
    starting_state = ret._starting

    # check that all transitions have sums of probabilities that equate to 1
    added_silence = False
    for state in ret._states:
        prob_sum = 0.0

        for letter in ret._alphabet:
            prob_sum += ret.probability(state, letter)

        # check if we need to add a silent transition
        if abs(1 - prob_sum) > 0.01:
            moves = ret._transitions[state]
            counts = ret._counting[state]

            missing = 1.0 - prob_sum
            total_weight = sum(counts.values())
            if total_weight == 0:
                total_weight = 1.0

            moves[SILENT_SYMBOL] = starting_state
            counts[SILENT_SYMBOL] = missing * total_weight
            added_silence = True

    # check if we need to update the alphabet
    if added_silence:
        ret._alphabet.add(SILENT_SYMBOL)

    return ret


def intersect_nets(
    left_net: FiniteLabelledMarkovChain,
    right_net: FiniteLabelledMarkovChain,
    only_keep_traversable_states: bool = True,
    keep_dead_states: bool = False,
    allow_optimisation: bool = True,
) -> IntersectedFiniteLabelledMarkovChain:
    """
    Computes and returns the intersected FLMCs between the given FLMCs.
    """
    # shared_alphabet = left_net._alphabet.union(right_net._alphabet)

    # totally_left_net = convert_net_to_totally_defined(left_net, shared_alphabet)
    # totally_right_net = convert_net_to_totally_defined(right_net, shared_alphabet)

    ret = IntersectedFiniteLabelledMarkovChain(
        left_net,
        right_net,
        keep_traversable=only_keep_traversable_states,
        keep_dead_states=keep_dead_states,
        allow_optimisation=allow_optimisation
    )

    return ret


def walk_and_assign_identifies(
    net: FiniteLabelledMarkovChain | IntersectedFiniteLabelledMarkovChain,
) -> Dict[MarkovState | IntersectedMarkovState, int]:
    """
    Walks the net and returns a mapping to assign identifer to each state.
    s_0 is assigned to the initial state of the net.
    """
    ret = {}
    identifier = 0

    traversed = set()
    queue = [net._starting]

    while len(queue) > 0:
        curr = queue.pop(0)

        if curr in traversed:
            continue

        traversed.add(curr)
        ret[curr] = identifier
        identifier += 1

        if curr not in net._transitions:
            print(f"could not find transitions for {curr=}")
            continue

        temp = []
        for state in net._transitions[curr].values():
            if state not in traversed:
                temp.append(state)
        temp = sorted(temp, key=lambda state: state.dot_label())

        queue.extend(temp)

    return ret


def make_one_step_transition_probabilties(
    net: FiniteLabelledMarkovChain | IntersectedFiniteLabelledMarkovChain,
) -> Tuple[np.matrix, Dict[MarkovState | IntersectedMarkovState, int]]:
    """
    Computes and returnsthe probabilities matrix for one-step transitions.
    The identifies used for the matrix are also returned.
    """
    num_states = len(net._states)
    p_matrix = np.zeros((num_states, num_states))

    # walk net to make a deterministic listing of identifies.
    identifiers = walk_and_assign_identifies(net)

    for source in net._states:
        for action in net.actions_from(source):
            target = net.moves_to(source, action)

            s_id = identifiers[source]
            t_id = identifiers[target]

            p_matrix[s_id, t_id] = net.probability(source, action)

    return p_matrix, identifiers


def _compute_long_run_with_numpy(
    prob_matrix: np.ndarray,
    ids: Dict,
) -> Dict[int, "Rational"]:
    """
    Fallback solver using numpy linear algebra.
    Solves (P^T - I)π = 0 with Σπ_i = 1 by replacing the last row
    of (P^T - I) with the normalisation constraint.
    """
    length = prob_matrix.shape[0]
    A = prob_matrix.T - np.eye(length)
    b = np.zeros(length)
    A[-1, :] = 1.0
    b[-1] = 1.0

    pi = np.linalg.solve(A, b)

    ret = {}
    for id in range(length):
        ret[id] = Rational(pi[id]).limit_denominator(10**12)
    return ret


def compute_long_run_proportions(
    net: FiniteLabelledMarkovChain | IntersectedFiniteLabelledMarkovChain, retrys=3
) -> Tuple[Dict[MarkovState | IntersectedMarkovState, float], List[Eq]]:
    """
    Constructs the linear equations to compute the long-run proportions,
    usually denoted by pi. These long-run probabilties represent the proportion
    of time the stochastic process will remain in the states in the long run.

    Returns mapping from states to their long-run proportions and the equations
    used to solve for these proportions.

    Note:
        Attempts to solve the steady-state system symbolically via sympy
        first for exact results. If sympy leaves free variables or returns
        no solution, falls back to numpy linear algebra.
    """
    ret = dict()

    # compute identifiers
    prob_matrix, ids = make_one_step_transition_probabilties(net)
    length = prob_matrix.shape[0]
    flipped_ids = dict((id, state) for state, id in ids.items())

    # construct sympy equalities
    equations = []
    long_run_symbols = {}

    for state, id in ids.items():
        long_run_symbols[id] = Symbol(f"pi_{id}")

    # pi_j = sum_i(pi_i * P[i,j]) for all j, dropping the last (redundant)
    for col in range(length - 1):
        rhs = sum(
            long_run_symbols[row]
            * Rational(prob_matrix[row, col]).limit_denominator(10**24)
            for row in range(length)
        )
        equations.append(Eq(long_run_symbols[col], rhs))

    # normalisation: sum of all pi must equal 1
    equations.append(Eq(sum(long_run_symbols.values()), 1))

    # attempt sympy solve
    symbols_list = list(long_run_symbols[i] for i in range(length))
    info("solving equations in long-run :: " + repr(equations))
    solutions = linsolve(equations, symbols_list)
    solutions = dict(
        (symbol, sol)
        for symbol, sol in zip(symbols_list, [sol for sol in solutions][0])
    )

    # check if sympy produced a complete, fully-determined solution
    incomplete = False
    if not isinstance(solutions, dict) or len(solutions) != length:
        print(solutions)
        print(
            f"problem with returned :: {type(solutions)} or {len(solutions)=}!={length}"
        )
        incomplete = True
    else:
        for symbol in symbols_list:
            val = solutions.get(symbol)
            if val is None or (hasattr(val, "free_symbols") and val.free_symbols):
                print(f"solution is incomplete, contains a free symbol :: {val}")
                incomplete = True
                break

    if incomplete:
        if retrys > 0:
            print("Sympy solver incomplete - retrying")
            return compute_long_run_proportions(net, retrys=retrys - 1)
        else:
            print("Sympy solver incomplete — falling back to numpy.")
            numpy_solutions = _compute_long_run_with_numpy(prob_matrix, ids)
            for id in range(length):
                ret[flipped_ids[id]] = numpy_solutions[id]
    else:
        for id, symbol in long_run_symbols.items():
            ret[flipped_ids[id]] = solutions[symbol]

    return ret, equations


def make_latex_for_probabilties(
    matrix: np.matrix,
    identifiers: Dict[MarkovState | IntersectedMarkovState, int],
    rounding=2,
) -> str:
    """
    Returns a LaTeX representation of the probabilties matrix.
    """
    width, height = matrix.shape

    ret = "&{} "

    # add identity vector
    identifiers = list(identifiers.items())
    identifiers = sorted(identifiers, key=lambda p: p[1])

    col_width = 2 + rounding
    ret += "\n\t\\begin{matrix}\n\t\t"
    header_entries = []

    if len(identifiers) > 10:
        mat_identifiers = identifiers[:8] + [("", None)] + [("", "j")]
    else:
        mat_identifiers = identifiers

    # add top level j vector
    for _, id in mat_identifiers:
        if id != None:
            label = f"s_{{{id}}}"
        else:
            label = f"\,\,\\cdots"
        id_digits = len(str(id))
        num_thin = max(0, col_width - id_digits - 1)
        if id == None:
            num_thin += 2
        spacing = "\\ " * num_thin
        header_entries.append(f"{spacing}{label}{"\\," * (num_thin-1)} &\n")
    ret += "\t\t".join(header_entries)
    ret = ret[:-3]
    ret += "\n\t\\end{matrix}"

    ret += "\\nonumber\n\\\\\n\\textbf{P} =\n"

    # add left vector for i
    if len(identifiers) > 10:
        mat_identifiers = identifiers[:8] + [("", None)] + [("", "i")]
    else:
        mat_identifiers = identifiers

    ret += "\t\\begin{matrix}\n\t\t"
    header_entries = []
    for _, id in mat_identifiers:
        if id != None:
            label = f"s_{{{id}}}"
        else:
            label = f"\\cdots"
        header_entries.append(f"{label} \\\\\n")
    ret += "\t\t".join(header_entries)
    ret = ret[:-3] + "\\,"
    ret += "\n\t\\end{matrix}\n"

    # add the transition matrix
    ret += "&{}\t\\begin{bmatrix}\n"
    r_wid = list(range(width))
    r_hid = list(range(height))

    # work if we need to add a break as matrix has max line length of ten
    lp_wid = max(r_wid) + 1
    up_wid = max(r_wid) + 1
    added_wid = False
    if len(r_wid) > 10:
        lp_wid = r_wid[8]
        up_wid = r_wid[-1]

    # loop through rows
    for wid in range(width):
        if wid == lp_wid:
            ret += "\t\t"
            for hid in range(lp_wid):
                ret += " \\cdots &"
            ret += " \\ddots &"
            ret += " \\cdots "
            ret = ret[:-1] + "\\\\"
            ret += "\n"
            added_wid = False
            continue
        elif wid > lp_wid and wid < up_wid:
            continue

        ret += "\t\t"

        # add the prob of the transitioning or dots
        for hid in range(height):
            if hid >= lp_wid and hid < up_wid:
                if not added_wid:
                    added_wid = True
                    ret += " \\cdots &"
                continue

            ret += f" {matrix[wid, hid]:1.{rounding}f} &"

        ret = ret[:-1] + "\\\\"
        ret += "\n"
        added_wid = False

    # close transition matrix
    ret = ret[:-4]
    ret += "\n\t\\end{bmatrix}"
    return ret


def convet_net_to_dot(
    net: FiniteLabelledMarkovChain,
    rounding=3,
    rankdir: str = "LR",
    net_name="FLMC",
    mclimit=100,
    min_len=5,
    ranksep=0.3,
    node_fontsize=10,
    transition_fontsize=12,
    size=0.47,
    identifiers=None,
    file_name=None,
    directory=None,
) -> Digraph:
    """
    Creates a dot representation of the net, then visualises the net by
    producing a dot file and pdf file of the net in the same location, using
    the name of the net.

    Returns the `graphviz.Digraph` representation of the net.
    """

    digraph_body = f"""
    ranksep={ranksep};
    nodesep=0.10;
    rankdir={rankdir};
    mclimit={mclimit};
    minlen={min_len};
    dpi=300;
    pad=0.1;
    margin=0.1;
    node [width={size}, height={size}, fixedsize=true, fontname="sans-serif",fontsize={node_fontsize}];
    edge [arrowsize=0.7,arrowhead=vee];\n"""

    def proc(label: str) -> str:
        return label.replace("<", "&lt;").replace(">", "&gt;")

    dot = Digraph(net_name, engine="dot", body=digraph_body)

    top_bot = rankdir != "LR"

    for fstate in net._accepting:
        if fstate != net._starting:
            dot.node(
                proc(str(fstate)),
                shape="doublecircle" if fstate.dot_shape() != "Mrecord" else "Mrecord",
                style="filled",
                colorscheme="set28",
                fillcolor="2",
                label=fstate.dot_label(top_bot),
                xlabel=f"s{identifiers[fstate]}" if identifiers else "",
                penwidth="1" if fstate.dot_shape() != "Mrecord" else "3",
            )
        else:
            dot.node(
                proc(str(fstate)),
                shape="doublecircle" if fstate.dot_shape() != "Mrecord" else "Mrecord",
                style="filled",
                colorscheme="set28",
                fillcolor="1",
                label=fstate.dot_label(top_bot),
                xlabel=f"s{identifiers[fstate]}" if identifiers else "",
                penwidth="1" if fstate.dot_shape() != "Mrecord" else "3",
            )

    for state in set(net._states).difference(set(net._accepting)):
        if (
            isinstance(state, IntersectedMarkovState)
            and state.left == DEAD_STATE
            and state.right == DEAD_STATE
        ):
            dot.node(
                proc(str(state)),
                shape="circle" if state.dot_shape() != "Mrecord" else "Mrecord",
                style="filled",
                colorscheme="set28",
                fillcolor="8",
                label=state.dot_label(top_bot),
                xlabel=f"s{identifiers[state]}" if identifiers else "",
            )
        elif state == DEAD_STATE:
            dot.node(
                proc(str(state)),
                shape="circle" if state.dot_shape() != "Mrecord" else "Mrecord",
                style="filled",
                colorscheme="set28",
                fillcolor="8",
                label=state.dot_label(top_bot),
                xlabel=f"s{identifiers[state]}" if identifiers else "",
                penwidth="1" if state.dot_shape() != "Mrecord" else "3",
            )
        elif state != net._starting:
            dot.node(
                proc(str(state)),
                shape=state.dot_shape(),
                label=state.dot_label(top_bot),
                xlabel=f"s{identifiers[state]}" if identifiers else "",
            )
        else:
            dot.node(
                proc(str(state)),
                shape=state.dot_shape(),
                style="filled",
                colorscheme="set28",
                fillcolor="1",
                xlabel=f"s{identifiers[state]}" if identifiers else "",
                label=state.dot_label(top_bot),
            )

    dot.node(
        "start", shape="point", width="0.1", height="0.1", xlabel="start", label=""
    )
    dot.edge("start", proc(str(net._starting)))

    for state in net._states:
        for symbol in net.actions_from(state):

            try:
                end = net.moves_to(state, symbol)
                prob = net.probability(state, symbol)
                # activity node
                scaler = 12.0 / transition_fontsize
                width = 0.6 / scaler
                height = 0.4 / scaler
                dot.node(
                    f"{proc(str(state))}_{symbol}",
                    label=(
                        f"{symbol} | {round(prob,rounding)}"
                        if prob > 0
                        else f"{symbol}"
                    ),
                    fontsize=f"{transition_fontsize}",
                    width=f"{width}" if prob > 0 else "0.25",
                    height=f"{height}" if prob > 0 else "0.25",
                    shape="plaintext",
                    fontname="Times-Roman",
                    style="rounded,dashed",
                    colorscheme="set28",
                    fillcolor="7" if prob > 0 else "8",
                    fontcolor="1" if prob > 0 else "8",
                )

                # add connecting arc to the
                if state != end:
                    dot.edge(
                        proc(str(state)),
                        f"{proc(str(state))}_{symbol}",
                        dir="none",
                    )
                    dot.edge(f"{proc(str(state))}_{symbol}", proc(str(end)))
                else:
                    dot.edge(
                        proc(str(state)),
                        f"{proc(str(state))}_{symbol}",
                        dir="both",
                    )
            except ValueError:
                continue
            except Exception as e:
                print(f"Unexpected error :: {e}")
    dot.render(filename=file_name, directory=directory)
    return dot


def convert_net_to_tikz(net: FiniteLabelledMarkovChain) -> str:
    """
    Returns the converted representation of the net in tikz for LaTeX.
    """

    def format_label(label: str) -> str:
        """Format a label for TikZ display, handling special characters"""
        label_str = str(label)
        # Handle empty trace
        if label_str == EMPTY_TRACE_SYMBOL or label_str == "":
            return r"{\tiny \emptytrace}"
        # Replace < and > with \langle and \rangle, and wrap in \tiny
        label_str = label_str.replace("<", r"\langle ").replace(">", r" \rangle")
        return r"{\tiny " + label_str + r"}"

    # Build graph structure to determine layout
    state_to_id = {}
    positioned_states = []

    # Start with the initial state
    initial_state = net._starting
    state_to_id[initial_state] = "s1"
    positioned_states.append(
        (initial_state, None, None)
    )  # (state, position_type, reference_node)

    # Find epsilon state and place it above initial
    epsilon_state = None
    for state in net._states:
        if str(state) == EMPTY_TRACE_SYMBOL:
            epsilon_state = state
            state_to_id[state] = "s2"
            positioned_states.append((state, "above", "s1"))
            break

    # Build a queue for BFS traversal from initial state
    from collections import deque

    visited = {initial_state}
    if epsilon_state:
        visited.add(epsilon_state)

    queue = deque([initial_state])
    counter = 3 if epsilon_state else 2

    while queue:
        current_state = queue.popleft()
        current_id = state_to_id[current_state]

        # Get all outgoing transitions from current state
        outgoing = []
        for symbol in net.actions_from(current_state):
            try:
                next_state = net.moves_to(current_state, symbol)
                if next_state not in visited:
                    outgoing.append(next_state)
            except (ValueError, Exception):
                continue

        # Position outgoing states
        for i, next_state in enumerate(outgoing):
            if next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)
                state_to_id[next_state] = f"s{counter}"

                if i == 0:
                    # First outgoing edge - place to the right
                    positioned_states.append((next_state, "right", current_id))
                else:
                    # Additional outgoing edges - place above the first one
                    first_outgoing_id = state_to_id[outgoing[0]]
                    positioned_states.append((next_state, "above", first_outgoing_id))

                counter += 1

    # Add any remaining unvisited states (shouldn't happen in well-formed graph)
    for state in net._states:
        if state not in visited:
            state_to_id[state] = f"s{counter}"
            if positioned_states:
                last_id = positioned_states[-1][0]
                last_node_id = state_to_id[last_id]
                positioned_states.append((state, "right", last_node_id))
            else:
                positioned_states.append((state, None, None))
            counter += 1

    # Start TikZ figure
    ret = r"\begin{figure}[tb]" + "\n"
    ret += r"    \centering" + "\n"
    ret += r"    \begin{tikzpicture}[->, >=Stealth, node distance=1cm]" + "\n"

    # Generate nodes with calculated positions
    for state, position_type, reference in positioned_states:
        node_id = state_to_id[state]
        label = format_label(str(state))

        # Build node attributes
        attrs = ["state"]

        # Check if initial state
        if state == net._starting:
            attrs.append("initial")

        # Check if accepting state
        if state in net._accepting:
            attrs.append("accepting")

        # Determine position
        if position_type and reference:
            position = f", {position_type}=of {reference}"
        else:
            position = ""

        node_def = (
            f"      \\node[{', '.join(attrs)}{position}] ({node_id}) {{${label}$}};"
        )
        ret += node_def + "\n"

    ret += "    \n"
    ret += r"      \draw" + "\n"

    # Generate edges
    edges = []
    for state in net._states:
        state_id = state_to_id[state]

        for symbol in net.actions_from(state):
            try:
                end_state = net.moves_to(state, symbol)
                end_id = state_to_id[end_state]
                prob = net.probability(state, symbol)
                symbol_label = format_label(symbol)

                # Format edge with probability
                edge_str = f"            ({state_id}) edge node[above] {{${symbol_label}, {prob:.2g}$}} ({end_id})"
                edges.append(edge_str)
            except (ValueError, Exception):
                continue

    # Join all edges with newlines, ending with semicolon
    if edges:
        ret += "\n".join(edges) + ";\n"
    else:
        ret += ";\n"

    ret += r"    \end{tikzpicture}" + "\n"
    ret += r"    \caption{FLMC representation}" + "\n"
    ret += r"    \label{fig:flmc}" + "\n"
    ret += r"\end{figure}" + "\n"

    return ret
