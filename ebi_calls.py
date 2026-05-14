from os.path import join
import subprocess
from typing import List

from sympy import Rational

from textx import metamodel_from_file
from textx.metamodel import TextXMetaModel

from pmkoalas.models.petrinets.wpn import (
    Place,
    WeightedTransition,
    PetriNetMarking,
    WeightedAcceptingPetriNet,
    BuildablePetriNet,
)

EBI_FOLD = join(".", "ebi")
EBI_BIN = join(EBI_FOLD, "ebi-binary.exe")
SLPN_TX = join(EBI_FOLD, "slpn.tx")


class PlainText:
    """
    Grammer metamodel class for plain text in SLPNs.
    """

    def __init__(self, parent, text) -> None:
        self.text: str = text


class Weight:
    """
    Grammer metamodel class for weights of transitions.
    """

    def __init__(self, parent, left, right, weight) -> None:
        if left is not None and len(left) and right is not None and len(right):
            self.value = Rational(f"{left}/{right}")
        else:
            self.value = Rational(f"{weight}")


class TransitionLabel:
    """
    Grammar metamodel class for labels of transitions
    """

    tau_number = 0

    def __init__(self, parent, label: PlainText) -> None:
        if label is not None and len(label.text):
            self.is_silent = False
            self.label = label.text
        else:
            self.label = f"tau_{TransitionLabel.tau_number:02d}"
            TransitionLabel.tau_number += 1
            self.is_silent = True


class Transition:
    """
    Grammar metamodel class for transitions in SLPNs.
    """

    def __init__(self, parent, id, label, weight, num_inputs, inputs, num_outputs, outputs) -> None:
        self.parent = parent
        self.id: int = id
        self.label: TransitionLabel = label
        self.weight: Weight = weight
        self.num_inputs: int = num_inputs
        self.inputs: List[int] = inputs
        self.num_outputs: int = num_outputs
        self.outputs: List[int] = outputs


class SLPN:
    """
    Grammar metamodel for SLPN.
    """

    def __init__(
        self, net_name, num_places, marked, num_transitions, transitions
    ) -> None:
        self.net_name: PlainText = net_name
        self.num_places: int = num_places
        self.marked: List[int] = marked
        self.num_transitions: int = num_transitions
        self.transitions: List[Transition] = transitions

    def convert_to_net(self) -> WeightedAcceptingPetriNet:
        """
        Converts the slpn from ebi into a wieghted net from pmkoalas.
        """
        ini_marking = {}
        places = {}
        builder = BuildablePetriNet()
        print(self.marked, self.num_places)
        # add places
        for place_id, marks in zip(range(self.num_places), self.marked):
            place = Place(f"p_{place_id:02d}")
            ini_marking[place] = marks
            builder.add_place(place)
            places[place_id] = place

        # add transitions with weights
        for transition in self.transitions:
            trans = WeightedTransition(
                transition.label.label,
                silent=transition.label.is_silent,
                weight=transition.weight.value,
            )
            builder.add_transition(trans)

            # add arcs
            for input in transition.inputs:
                builder.add_arc_between(
                    places[input],
                    trans
                )
            for output in transition.outputs:
                builder.add_arc_between(
                    trans,
                    places[output]
                )

        # find places without any outgoing transitions
        slpn = builder.create_net()
        fmarks = []
        for place in slpn.places:
            if len(slpn.postset(place)) == 0:
                fmarks.append(
                    PetriNetMarking(
                        {
                            place: 1
                        }
                    )
                )

        return WeightedAcceptingPetriNet(
            slpn, PetriNetMarking(ini_marking), fmarks
        )


def metamodel_for_slpn() -> TextXMetaModel:
    """
    Gets the grammar for a slpn, ready to parse an slpn file
    from ebi.
    """
    return metamodel_from_file(
        SLPN_TX, classes=[SLPN, Transition, TransitionLabel, PlainText, Weight]
    )


def parse_slpn(filepath: str) -> SLPN:
    """
    Parses the given .slpn from ebi into a readable grammar form.
    """
    metamodel = metamodel_for_slpn()
    return metamodel.model_from_file(filepath)


def parse_to_weighted_net(filepath: str) -> WeightedAcceptingPetriNet:
    """
    Attempts to parse the given .slpn from ebi into pmkoalas net.
    """
    domain = parse_slpn(filepath)
    return domain.convert_to_net()


def discover_occurrence_stochastic_labelled_petri_net(log_path, net_path, out_path):
    """
    Calls ebi to discover an occurence slpn.
    """
    alias = [EBI_BIN, "disc", "occ", "slpn"]
    ret = subprocess.run(
        alias + [log_path, net_path, "-o", out_path], capture_output=True, text=True
    )

    print("STDOUT ::")
    print(ret.stdout)
    if ret.returncode != 0:
        print("STDERR ::")
        print(ret.stderr)
    else:
        ret = parse_to_weighted_net(out_path)

    return ret


if __name__ == "__main__":
    from pmkoalas.models.dotutil import lpn_prettier_dot

    ret = parse_to_weighted_net("foo.slpn")

    with open("testing.dot", "w") as f:
        f.write(lpn_prettier_dot(ret))

    from petri_nets.marking_system import MarkingSystem

    ms = MarkingSystem(ret)

    print(repr(ms))

    with open("ms.dot", "w") as f:
        f.write(ms.to_dot())
