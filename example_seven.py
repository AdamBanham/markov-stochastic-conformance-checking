from markov import (
    FiniteLabelledMarkovChain,
    MarkovState,
    make_one_step_transition_probabilties,
    make_latex_for_probabilties,
    compute_long_run_proportions,
)
from pmkoalas._logging import setLevel
setLevel("INFO")


def create():
    from os.path import join
    from util import clear_directory

    dump_directory = join(".", "sample_nets", "example_07")
    clear_directory(dump_directory)

    chain = FiniteLabelledMarkovChain()
    chain._alphabet.add("upper")
    chain._alphabet.add("middle")
    chain._alphabet.add("lower")

    # add upper
    chain._states = set()
    chain._states.add(MarkovState(["upper"]))
    chain._starting = MarkovState(["upper"])
    chain._transitions[MarkovState(["upper"])] = {
        "upper": MarkovState(["upper"]),
        "middle": MarkovState(["middle"]),
        "lower": MarkovState(["lower"]),
    }
    chain._counting[MarkovState(["upper"])] = {
        "upper": 0.45,
        "middle": 0.48,
        "lower": 0.07,
    }

    # add middle
    chain._states.add(MarkovState(["middle"]))
    chain._transitions[MarkovState(["middle"])] = {
        "upper": MarkovState(["upper"]),
        "middle": MarkovState(["middle"]),
        "lower": MarkovState(["lower"]),
    }
    chain._counting[MarkovState(["middle"])] = {
        "upper": 0.05,
        "middle": 0.70,
        "lower": 0.25,
    }

    # add lower
    chain._states.add(MarkovState(["lower"]))
    chain._transitions[MarkovState(["lower"])] = {
        "upper": MarkovState(["upper"]),
        "middle": MarkovState(["middle"]),
        "lower": MarkovState(["lower"]),
    }
    chain._counting[MarkovState(["lower"])] = {
        "upper": 0.01,
        "middle": 0.50,
        "lower": 0.49,
    }

    prob, id = make_one_step_transition_probabilties(chain)
    print(id)
    with open(join(dump_directory, "probs.latex"), "w") as f:
        f.write(make_latex_for_probabilties(prob, id))
        f.write("\n\n")
        f.write(repr(id))

    long_runs, eqs = compute_long_run_proportions(chain)

    with open(join(dump_directory, "equations.out"), "w") as f:
        for key, val in long_runs.items():
            f.write(f"{key} -> {float(val)}\n")
        f.write("\n\n")
        for eq in eqs:
            f.write(repr(eq) + "\n")


if __name__ == "__main__":
    create()
