from markov import FiniteLabelledMarkovChain
from pmkoalas.simple import EventLog
from pmkoalas.complex import ComplexEventLog
from pmkoalas.models.petrinets.wpn import WeightedAcceptingPetriNet


def discover_chain_from_log(
    log: EventLog, window_length: int = -1
) -> FiniteLabelledMarkovChain:
    """
    Discovers a chain for the given log by walking over the variants
    of the log.

    By default the discover chain will be the prefix tree of the log,
    but a window size can be passed to simplify the chain, where only
    the last n-steps are considered for defining states in the chain.
    """
    net = FiniteLabelledMarkovChain()

    for variant, freq in log:
        net.add_variant(variant, freq, window_length=window_length)

    return net


def _testing_chain_discovery():
    from pmkoalas.dtlog import convert
    from markov import convet_net_to_dot, convert_net_to_totally_defined
    from markov import convert_net_to_be_recurrent
    from util import clear_directory
    from os.path import join

    dump_directory = join(".", "sample_nets", "discovering")
    clear_directory(dump_directory)

    traces = (
        [""]
        + ["a f"] * 2
        + ["a b"] * 6
        + ["a b c"] * 6
        + ["a c b"] * 7
        + ["a c b c f"] * 7
        + ["a c b b y"] * 3
    )
    log = convert(*traces)

    for win in range(1, 4):
        chain = discover_chain_from_log(log, window_length=win)
        _ = convet_net_to_dot(
            chain, net_name=f"win_len_{win}", directory=dump_directory
        )
        t_chain = convert_net_to_totally_defined(chain)
        _ = convet_net_to_dot(
            t_chain, net_name=f"tot_win_len_{win}", directory=dump_directory
        )
        recurrent_chain = convert_net_to_be_recurrent(t_chain)
        _ = convet_net_to_dot(
            recurrent_chain, net_name=f"rec_win_len_{win}", directory=dump_directory
        )


def discover_stochastic_net(log: ComplexEventLog):
    """
    Using the inductive miner, computes a control-flow model and then
    discovers a weights for the net.
    """
    from pmkoalas.export import export_to_xes_complex
    from tempfile import TemporaryFile
    from pm4py import read_xes, discover_petri_net_inductive, write_pnml
    from ebi_calls import discover_occurrence_stochastic_labelled_petri_net

    net_file = TemporaryFile("w", suffix=".pnml")
    net_file.close()
    log_file = TemporaryFile("w", suffix=".xes")
    log_file.close()
    slpn_file = TemporaryFile("w", suffix=".slpn")
    slpn_file.close()

    # create temp file for transfer to pm4py
    export_to_xes_complex(log_file.name, log)

    # discover inductive net
    plog = read_xes(log_file.name)
    net, im, fm = discover_petri_net_inductive(
        plog, 
        noise_threshold=0.20
    )
    write_pnml(net, im, fm, net_file.name)
    write_pnml(net, im, fm, "dead_road_fines.pnml")

    # discover with ebi-pm
    _ = discover_occurrence_stochastic_labelled_petri_net(
        log_file.name,
        net_file.name,
        slpn_file.name
    )

    with open(join(".", "foo.slpn"), "w") as f:
        result = open(slpn_file.name, "r").read()
        f.write(result)

    return result


def convert_net_to_chain(net:WeightedAcceptingPetriNet) -> FiniteLabelledMarkovChain:
    """
    Converts a weighted petri net into a FLMC.
    """
    pass


if __name__ == "__main__":
    from os.path import join
    from pmkoalas.read import read_xes_complex
    from pmkoalas._logging import setLevel
    setLevel("INFO")

    LOG_FILE = join(".", "logs", "road_fines.xes")
    log = read_xes_complex(LOG_FILE) 

    ret = discover_stochastic_net(log)
    print(ret)
    print(repr(ret))
