from dataclasses import dataclass, field
from uuid import uuid4
from copy import deepcopy
from typing import Generic, List, TypeVar
from abc import abstractmethod
from functools import lru_cache

from pmkoalas.models.petrinets.wpn import (
    WeightedAcceptingPetriNet,
    WeightedPetriNetSemantics,
)

PLACE_COLOUR = "#ffe0b2"
START_PLACE_COLOUR = "#c5e1a5"
FINAL_PLACE_COLOUR = "#ef9a9a"


@dataclass
class Node:
    name: str
    id: str = field(default_factory=lambda: uuid4().hex)
    starting: bool = False
    final: bool = False

    def __hash__(self) -> int:
        return hash(id)

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Node):
            return self.name == value.name and self.id == self.id
        return False


@dataclass
class Vertex:
    source: str
    target: str
    label: str = ""
    weight: str | float = 1.0

    def __hash__(self) -> int:
        return hash((self.source, self.target))

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Vertex):
            return (
                value.label == self.label
                and value.source == self.source
                and value.target == self.target
            )
        return False


Nodes = TypeVar("Nodes", bound=Node)
Vertices = TypeVar("Vertices", bound=Vertex)


class Graph(Generic[Nodes, Vertices]):
    """
    Base class for most graphs.
    """

    def __init__(self, nodes: List[Nodes], vertices: List[Vertices]) -> None:
        self.nodes = deepcopy(nodes)
        self.vertices = deepcopy(vertices)

        self._node_map = dict((node.id, node) for node in self.nodes)

    @abstractmethod
    def postset(self, node: Nodes) -> List[Nodes]:
        """
        Returns the next nodes from the given node.
        """
        pass

    @abstractmethod
    def preset(self, node: Nodes) -> List[Nodes]:
        """
        Returns the previous nodes from the given node.
        """
        pass


class MarkingSystem(Graph[Node, Vertex]):
    """
    Represents a marking system of a Petri net.
    """

    def __init__(self, net: WeightedAcceptingPetriNet) -> None:
        nodes = {}
        vertices = []

        def make_node_name(mark) -> str:
            return repr(mark)

        def make_node(mark) -> Node:
            semantics = WeightedPetriNetSemantics(net, mark)

            return Node(make_node_name(mark), final=len(semantics.can_fire()) == 0)

        def find_node(mark) -> Node:
            if make_node_name(mark) not in nodes:
                node = make_node(mark)
                nodes[node.name] = node
            return nodes[make_node_name(mark)]

        # walk from the initial marking
        queue = [net.initial_marking]
        seen = set()
        while len(queue) > 0:
            print(f"making system :: {len(queue)=}, {len(seen)=}", end="\r")
            curr = queue.pop(0)

            if curr in seen:
                continue
            else:
                seen.add(curr)

            node = find_node(curr)
            if curr == net.initial_marking:
                node.starting = True

            semantics = WeightedPetriNetSemantics(net, curr)
            for transition in semantics.fireable():
                peek = semantics.peek(transition)

                if peek._curr not in seen:
                    queue.append(peek._curr)
                    target = find_node(peek._curr)
                    vertices.append(
                        Vertex(node.id, target.id, transition.name, transition.weight)
                    )

        super().__init__(list(nodes.values()), vertices)

    @lru_cache
    def postset(self, node: Node) -> List[Node]:
        """
        Returns the next nodes from the given node.
        """
        ret = []
        for vertex in self.vertices:
            if vertex.source == node.id:
                ret.append(self._node_map[vertex.target])
        return ret

    @lru_cache
    def preset(self, node: Nodes) -> List[Nodes]:
        """
        Returns the previous nodes from the given node.
        """
        ret = []
        for vertex in self.vertices:
            if vertex.target == node.id:
                ret.append(self._node_map[vertex.source])
        return ret

    def to_dot(self) -> str:
        """
        Returns a formatted literal string for the graphviz dot representation
        of the marking system.
        """

        def esc(value: object) -> str:
            return str(value).replace("\\", "\\\\").replace('"', '\\"')

        ret = "digraph{\n"
        ret += "\tdpi=150;rankdir=LR;nodesep=0.6;ranksep=0.3;\n"
        ret += '\tedge[penwidth=4,fontsize=16,minlen=2,fontname="roboto"];\n'
        ret += f'\t\tnode[shape=circle,margin="0.05, 0.05",style=filled,width=1,penwidth=2,fontsize=16,fontname="roboto"];\n'

        node_ids: dict[str, str] = {}
        for idx, node in enumerate(self.nodes):
            dot_id = f"n{idx}"
            node_ids[node.id] = dot_id

            node_colour = PLACE_COLOUR
            if node.starting:
                node_colour = START_PLACE_COLOUR
            elif node.final:
                node_colour = FINAL_PLACE_COLOUR

            labeler = esc(node.name)
            label = (
                "<"
                + "<BR/>".join(
                    [labeler[i : i + 16] for i in range(0, len(labeler), 16)]
                )
                + ">"
            )
            ret += f'\t{dot_id}[label={label}, fillcolor="{node_colour}"];\n'

        for edge_idx, vertex in enumerate(self.vertices):
            source = node_ids.get(vertex.source)
            target = node_ids.get(vertex.target)
            if source is None or target is None:
                continue

            if vertex.label:
                edge_label = f"{vertex.label} ({vertex.weight})"
            else:
                edge_label = str(vertex.weight)

            label_node = f"e{edge_idx}"
            ret += (
                f'\t{label_node}[label="{esc(edge_label)}", '
                "shape=plain, style=solid, margin=\"0.05, 0.05\", width=0, "
                "height=0, penwidth=0, fontsize=16, "
                'fontname="roboto"];\n'
            )
            ret += f"\t{source} -> {label_node}[arrowhead=none, minlen=1];\n"
            ret += f"\t{label_node} -> {target}[minlen=1];\n"

        ret += "}\n"
        return ret
