# Steady-State Analysis: Stochastic Conformance Checking with Guarantees

This repository includes an implementation of the proposed stochastic
conformance measures within the paper.
It also includes the evaluation set up used within the paper.

## Event logs

In order to run the evaluation, make sure to add the event logs with the
outlined names in the `logs` directory of the repository. See the event log
[readme.md](./logs/readme.md) for more information.

## Development Environment

The evaluation used a Python `3.13.x` interpreter and the environment can be
reconstructed using `pipenv`.

Once `pipenv` has been installed and you have a `3.13.x` interpreter installed,
the following commands will produce the same virtual enviornment in our
evaulation.

```bash
python -m pipenv install
python -m pipenv sync
python -m pipenv shell
```

The following commands all assume that you have activated the virtual 
environment, i.e. `python -m pipenv shell`, before following the commands.

## Evaluation

Figure 5. can be recomputed using the following commands:

```bash
python evaluation_shift_by_one.py
# once done
python plot_shift_by_one.py
```

Figure 6. can be recomputed using the following commands:

```bash
python evaluation_runtime.py
# once done
python plot_runtime.py
```

## Figure Reproductions

The underlying dot graphviz files for figures used in the paper can be found
in `sample_nets` and typically have a script at the root of the directory
for reproducing the graphs following the same name as the folder within
`sample_nets`.


