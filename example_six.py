from measure import compute_stochastic_entropy_precision
from measure import compute_stochastic_entropy_recall
from pmkoalas.dtlog import convert
from pmkoalas._logging import setLevel, info
from logging import INFO

setLevel(INFO)

from json import dumps


def create():

    traces = [""] + ["a f"] * 6 + ["a b"] * 5 + ["a b c"] * 5 + ["a c b"] * 6
    a_log = convert(*traces)

    other_traces = ["a b"] * 6 + ["a b c"] * 6 + ["a c b"] * 7

    # build sample curve
    recalls = []
    precisions = []
    other_logs = []

    for sample_length in range(1, len(other_traces)+1):
        info("*" * 10)
        info(f"starting sample :: {sample_length}")
        info("*" * 10)
        # build other log
        traces = other_traces[:sample_length]
        b_log = convert(*traces)

        other_logs.append(str(b_log))

        # compute recall for a_net and b_net
        recall = compute_stochastic_entropy_recall(
            a_log,
            b_log,
        )

        # precision of A on B
        precision = compute_stochastic_entropy_precision(
            a_log,
            b_log,
        )

        recalls.append(recall)
        precisions.append(precision)

    with open("example_six_curves.json", "w") as f:
        f.write(
            dumps(
                {
                    "a_log": str(a_log),
                    "b_logs": other_logs,
                    "recall": recalls,
                    "precision": precisions,
                },
                indent=4,
            )
        )


if __name__ == "__main__":
    create()
