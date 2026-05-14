import numpy as np
import graphviz
import json
import os
import copy


class FLMC:
    """
    The class FLMC (Finite Labelled Markov Chain)
    ...
    Attributes
    ==========
    name : string
        The name of the Finite Labelled Markov Chain.
    states : list
        A list of the states in the Finite Labelled
        Markov Chain.
    init_state : string
        The initial state of the Finite Labelled
        Markov Chain.
    accepting_states : list
        The accepting, or final, states in the Finite
        Labelled Markov Chain.
    symbols : list
        The alphabet of symbols transitioned on by the
        Finite Labelled Markov Chain.
    prob_func_mat : dict
        The probability of transition from a given
        state on a particular symbol. This should
        be defined in the form of a dict with
        states as keys and more dicts as values.
        These dicts should have symbols as keys and
        probabilities as values (and the sum of the
        values in each dict should be 1).
    trans_func_mat : dict
        The transition function defining the outputted
        state for a transition from a given state
        on a particular symbol. This should be
        defined in the same way as prob_func_mat,
        only instead of probabilties, specify the
        outputted state. The outputted state for a
        transition with probability 0 does not
        matter, and can be an empty string.

    Methods
    =======
    check()
        A function to check that the Finite Labelled
        Markov Chain satisfies the requirements to
        be a Finite Labelled Markov Chain.
    viz()
        A function to display the Finite Labelled
        Markov Chain.
    limiting_probs()
        A function that calculates the limiting
        probabilities of the Finite Labelled Markov
        Chain.
    export()
        A function to export the FLMC to a json file.
    """

    def __init__(
        self,
        name,
        states,
        init_state,
        accepting_states,
        symbols,
        prob_func_mat,
        trans_func_mat,
    ):
        """
        The initialisation function to create an
        object of class FLMC.

        Parameters
        ==========
        name : string
            The name of the Finite Labelled Markov
            Chain.
        states : list
            A list of the states in the Finite
            Labelled Markov Chain.
        init_state : string
            The initial state of the Finite Labelled
            Markov Chain.
        accepting_states : list
            The accepting, or final, states in the
            Finite Labelled Markov Chain.
        symbols : list
            The alphabet of symbols transitioned on by
            the Finite Labelled Markov Chain.
        prob_func_mat : dict
            The probability of transition from a given
            state on a particular symbol. This
            should be defined in the form of a dict
            with states as keys and more dicts as
            values. These dicts should have symbols
            as keys and probabilities as values
            (and the sum of the values in each dict
            should be 1).
        trans_func_mat : dict
            The transition function defining the
            outputted state for a transition from a
            given state on a particular symbol.
            This should be defined in the same way
            as prob_func_mat, only instead of
            probabilties, specify the outputted
            state. The outputted state for a
            transition with probability 0 does not
            matter, and can be an empty string.

        Returns
        =======
        None.
        """
        self.name = name
        self.states = states
        self.init_state = init_state
        self.final_states = accepting_states
        self.symbols = symbols
        self.prob_func = prob_func_mat
        self.trans_func = trans_func_mat

    def check(self):
        """
        A function to check that the Finite Labelled
        Markov Chain satisfies the requirements to
        be a Finite Labelled Markov Chain.

        Returns
        =======
        bool
            Whether the Finite Labelled Markov Chain
            passes all checks or not.
        string
            If the Finite Labelled Markov Chain does
            not pass a check, then the reason for
            not passing the check.
        """
        if self.init_state not in self.states:
            return False, "Initial state not in set of states"

        for f in self.final_states:
            if f not in self.states:
                return False, "Final state {} not in set of states".format(f)

        if len(self.prob_func.keys()) != len(self.states):
            return (
                False,
                "First dimension of symbol probability matrix does not have "
                "the same size as the number of states",
            )

        for i in range(len(self.prob_func.keys())):
            if len(self.prob_func[list(self.prob_func.keys())[i]].keys()) != len(
                self.symbols
            ):
                return (
                    False,
                    "Second dimension of symbol probability matrix does not "
                    "have the same size as the number of symbols",
                )

        if len(self.trans_func.keys()) != len(self.states):
            return (
                False,
                "First dimension of transition matrix does not have the same "
                "size as the number of states",
            )

        for i in range(len(self.trans_func.keys())):
            if len(self.trans_func[list(self.trans_func.keys())[i]].keys()) != len(
                self.symbols
            ):
                return (
                    False,
                    "Second dimension of transition matrix does not have the "
                    "same size as the number of symbols",
                )

        for index, key in enumerate(self.prob_func.keys()):
            if abs(sum(self.prob_func[key].values()) - 1) > (10**-5):
                return (
                    False,
                    "Row {} of symbol probability matrix does not sum to 1".format(
                        index
                    ),
                )

        for key in self.trans_func.keys():
            for j, element in enumerate(list(self.trans_func[key].values())):
                if element not in self.states:
                    return (
                        False,
                        (
                            "End state for transition from state {} on symbol {} "
                            / +"is not in the set of states"
                        ).format(key, self.symbols[j]),
                    )

        return True, "Yay!"

    def viz(self):
        """
        A function to display the Finite Labelled
        Markov Chain. The visualised Finite
        Labelled Markov Chain is outputted as a pdf
        file.

        Returns
        =======
        None.
        """
        dot = graphviz.Digraph(self.name)
        c, s = self.check()
        if not c:
            print(s)
            return

        for fstate in self.final_states:
            dot.node(str(fstate), shape="doublecircle")

        for state in set(self.states).difference(set(self.final_states)):
            dot.node(str(state), shape="circle")

        dot.node("", shape="point")
        dot.edge("", str(self.init_state))

        for start in self.states:
            for symbol in self.symbols:
                end = self.trans_func[start][symbol]
                if round(self.prob_func[start][symbol], 3) != 0:
                    dot.edge(
                        str(start),
                        str(end),
                        label="{}, {}".format(
                            symbol, round(self.prob_func[start][symbol], 3)
                        ),
                    )
        dot.view()

    def limiting_probs(self):
        """
        A function that calculates the limiting
        probabilities of the Finite Labelled Markov
        Chain.

        The limiting probabilities are the solution to
        the equation
        :math:`\\boldsymbol{\\pi}=\\boldsymbol{\\pi}M`,
        where :math:`M` is a probability matrix
        with elements :math:`\\left[M\\right]_{i,j}
        = \\sum_{a\\in\\sigma}P\\left(i,a\\right)`.

        Returns
        =======
        limiting : list of floats
            The limiting probabilities of the Finite
            Labelled Markov Chain.
        """
        # compute the incidence matrix M
        M = np.zeros((len(self.states), len(self.states)))
        for i, start_state in enumerate(self.states):
            for k, end_state in enumerate(self.states):
                sum_prob = 0
                for symbol in self.symbols:
                    this_prob = self.prob_func[start_state][symbol]
                    this_state = self.trans_func[start_state][symbol]
                    if this_state == end_state:
                        sum_prob += this_prob
                M[i, k] = sum_prob

        # do something?
        evals, evecs = np.linalg.eig(M.T)
        evec1 = evecs[:, np.isclose(evals, 1)]
        evec1 = evec1[:, 0]

        # normalise the limiting probabilities
        limiting = evec1 / evec1.sum()
        limiting = limiting.real
        return limiting

    def export(self, filename, working_directory):
        """
        A function to export a FLMC to a json file in
        the specified location (filename).

        Parameters
        ==========
        filename : string
            The filename to use to save the json file
            export of the FLMC model.
        working_directory : string, optional
            The filepath to the folder in which to
            save the json file.

        Returns
        =======
        None.
        """
        output_dict = {
            "name": self.name,
            "states": list(self.states),
            "init_state": self.init_state,
            "final_states": list(self.final_states),
            "symbols": list(self.symbols),
            "prob_func": self.prob_func,
            "trans_func": self.trans_func,
        }
        if not filename.lower().endswith(".json"):
            filename = filename + ".json"

        filepath = os.path.join(working_directory, "event_logs", filename)
        with open(filepath, "w") as f:
            json.dump(output_dict, f, indent=4)


def importFLMC(filepath):
    """
    A function to import a FLMC from the json file
    specified by the supplied filepath.

    Parameters
    ==========
    filepath : string
        The filepath to the json file containing the
        FLMC model.

    Returns
    =======
    newFLMC : FLMC
        the imported FLMC
    """
    with open(filepath, "r") as f:
        input_dict = json.load(f)

    name = input_dict["name"]
    states = input_dict["states"]
    init_state = input_dict["init_state"]
    final_states = input_dict["final_states"]
    symbols = input_dict["symbols"]
    prob_func = input_dict["prob_func"]
    trans_func = input_dict["trans_func"]

    newFLMC = FLMC(
        name, states, init_state, final_states, symbols, prob_func, trans_func
    )
    return newFLMC


def intersection(A: FLMC, B: FLMC, EMPTY_STRING="''", DEAD_STATE_NAME="DEAD"):
    """
    Finds the Finite Labelled Markov Chain that is the
    intersection of the Finite Labelled Markov
    Chains A and B.

    Parameters
    ==========
    A : an object of class FLMC
        A Finite Labelled Markov Chain
    B : an object of class FLMC
        A Finite Labelled Markov Chain
    EMPTY_STRING : string, optional
        An optional variable to handle cases where a
        transition on no symbol is needed. The
        default is "''".
    DEAD_STATE_NAME : string, optional
        An optional variable for handling cases where
        a dead state needs to be added. The default
        is "DEAD".

    Returns
    =======
    an object of class FLMC
        A Finite Labelled Markov Chain, the
        intersection of A and B
    """
    init_A = A.init_state
    final_A = A.final_states
    symbols_A = A.symbols
    probabilities_A = A.prob_func
    transitions_A = A.trans_func

    init_B = B.init_state
    final_B = B.final_states
    symbols_B = B.symbols
    probabilities_B = B.prob_func
    transitions_B = B.trans_func

    name_inter = A.name + " " + B.name
    symbols_inter = [s for s in symbols_A if s in symbols_B]
    init_inter = (init_A, init_B)
    states_inter = [init_inter]
    probs_inter = {}
    trans_inter = {}

    status = 1
    while status == 1:
        status = 0
        for state_AB in states_inter:
            if state_AB not in probs_inter.keys():
                probs_inter[state_AB] = {}
            if state_AB not in trans_inter.keys():
                trans_inter[state_AB] = {}

        states_inter_next = copy.deepcopy(states_inter)

        for state_AB in states_inter:
            state_A = state_AB[0]
            state_B = state_AB[1]

            for s in symbols_inter:
                out1 = transitions_A[state_A][s]
                out2 = transitions_B[state_B][s]
                prob1 = probabilities_A[state_A][s]
                prob2 = probabilities_B[state_B][s]
                prob_AB = min([prob1, prob2])
                out_AB = (out1, out2)

                if out_AB not in states_inter:
                    states_inter_next.append(out_AB)
                    status = 1

                probs_inter[state_AB][s] = prob_AB
                trans_inter[state_AB][s] = out_AB

        states_inter = states_inter_next

    final_inter = []
    for state_AB in states_inter:
        if (state_AB[0] in final_A) and (state_AB[1] in final_B):
            final_inter.append(state_AB)

    for start_inter in states_inter:
        row_sum = sum(probs_inter[start_inter].values())
        if row_sum != 0:
            for symb_inter in symbols_inter:
                probs_inter[start_inter][symb_inter] = (
                    probs_inter[start_inter][symb_inter] / row_sum
                )

    dead_state_needed = 0
    for start_inter in states_inter:
        row_sum = sum(probs_inter[start_inter].values())
        if row_sum != 1:
            dead_state_needed = 1

    if dead_state_needed == 1:
        for state in states_inter:
            if EMPTY_STRING not in list(probs_inter[state]):
                probs_inter[state][EMPTY_STRING] = 0
                trans_inter[state][EMPTY_STRING] = DEAD_STATE_NAME

        states_inter.append(DEAD_STATE_NAME)
        symbols_inter.append(EMPTY_STRING)
        symbols_inter = list(set(symbols_inter))
        probs_inter[DEAD_STATE_NAME] = {}
        trans_inter[DEAD_STATE_NAME] = {}
        probs_inter[DEAD_STATE_NAME][EMPTY_STRING] = 1
        trans_inter[DEAD_STATE_NAME][EMPTY_STRING] = DEAD_STATE_NAME

        for symb_inter in symbols_inter:
            probs_inter[DEAD_STATE_NAME][symb_inter] = 0
            trans_inter[DEAD_STATE_NAME][symb_inter] = DEAD_STATE_NAME

        for start_inter in states_inter:
            row_sum = sum(probs_inter[start_inter].values())
            if abs(row_sum - 1) >= 10**-6:
                probs_inter[start_inter][EMPTY_STRING] = 1 - row_sum
                trans_inter[start_inter][EMPTY_STRING] = DEAD_STATE_NAME

    return FLMC(
        name_inter,
        states_inter,
        init_inter,
        final_inter,
        symbols_inter,
        probs_inter,
        trans_inter,
    )


# The Python code to find the precision and recall of two FLMCs.
def delta(A: FLMC, B: FLMC):
    """
    Calculates
    ..math:: \\delta\\left(A,B\\right) = \\sum_{i\\in
    S_{A}}\\max\\left\\lbrace\\pi_{A,i}-\\sum_{j\\in
    S_{B}}\\pi_{AB,\\left(i\\times j\\right)},
    0\\right\\rbrace
    for FLMCs A and B, where :math:`\\pi_{AB}`
    indicates the limiting probabilities of the
    intersection of A and B.

    Parameters
    ==========
    A : class FLMC
        A Finite Labelled Markov Chain
    B : class FLMC
        A Finite Labelled Markov Chain

    Returns
    =======
    delta_value : float
    """
    C = intersection(A, B)
    pi_A = A.limiting_probs()
    pi_C = C.limiting_probs()

    delta_value = 0
    for i, state_A in enumerate(A.states):
        pi_A_i = pi_A[i]
        sum_over_B = 0
