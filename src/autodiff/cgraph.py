from __future__ import annotations

from typing import Dict, List, Set, Tuple

from ._cnode import ComputationalNode, Operation, Variable
from ._typing import NDArrayFloat
from .tensor_api import Tensor


class ComputationalGraph:

    """Handles assignment of operands, evaluation of computational graph and
    return of results.
    """

    _input_tensors: List[Tensor]
    _output_tensors: List[Tensor]
    _input_nodes: List[ComputationalNode]
    _output_nodes: List[ComputationalNode]
    _nodes: Set[ComputationalNode]
    _sorted_nodes: Tuple[ComputationalNode]
    _evaluation_order: Tuple[ComputationalNode]

    def __init__(
            self,
            input_tensors: List[Tensor],
            output_tensors: List[Tensor],
            /
            ) -> None:
        self._input_tensors = input_tensors
        self._output_tensors = output_tensors

    def compile(self, /) -> None:
        """Find evaluation order and validate computational graph."""
        self._input_nodes = list()
        for input_tensor in self._input_tensors:
            self._input_nodes += input_tensor.nodes

        self._output_nodes = list()
        for output_tensor in self._output_tensors:
            self._output_nodes += output_tensor.nodes

        # generate set of nodes
        self._nodes = set()
        node_stack = list(self._output_nodes)
        while node_stack:
            node = node_stack.pop()
            if node not in self._nodes:
                self._nodes.add(node)
                node_stack.extend(node.operands)

        # initialize in-degree counters
        in_degree_counters = dict.fromkeys(self._nodes, 0)
        for node in self._nodes:
            for operand_node in node.operands:
                in_degree_counters[operand_node] += 1

        # generate topological order
        topologically_sorted_nodes: List[ComputationalNode] = list()
        node_stack = list(output_node for output_node in self._output_nodes
                          if in_degree_counters[output_node] == 0)
        while node_stack:
            node = node_stack.pop()
            topologically_sorted_nodes.append(node)

            for operand_node in node.operands:
                in_degree_counters[operand_node] -= 1
                if in_degree_counters[operand_node] == 0:
                    node_stack.append(operand_node)

        self._sorted_nodes = tuple(reversed(topologically_sorted_nodes))
        self._evaluation_order = tuple(node for node in self._sorted_nodes
                                       if node.operands)

        # validate computational graph
        if not all(input_node in self._nodes
                   for input_node in self._input_nodes):
            raise ValueError('Input nodes are detached from computational '
                             'graph!')

    def __call__(self, *args: NDArrayFloat) -> Tuple[NDArrayFloat, ...]:
        """Evaluate computational graph."""
        if len(self._input_tensors) != len(args):
            raise ValueError(f'Number of provided ({len(args)}) and required '
                             f'({len(self._input_tensors)}) values are '
                             f'different!')

        for input_tensor, arg in zip(self._input_tensors, args):
            input_tensor.assign(arg)

        for node in self._evaluation_order:
            node.evaluate()

        result = tuple(output_tensor.access()
                       for output_tensor in self._output_tensors)

        return result

    def __copy__(self, /) -> ComputationalGraph:
        node_mapping: Dict[ComputationalNode, ComputationalNode] = dict()

        for node in self._sorted_nodes:
            if isinstance(node, Variable):
                copied_node = Variable(node.result)
            elif isinstance(node, Operation):
                copied_operands = tuple(node_mapping[operand] for operand
                                        in node.operands)
                copied_node = node.__class__(*copied_operands)
            else:
                raise NotImplementedError

            node_mapping[node] = copied_node

        copied_input_tensors: List[Tensor] = list()
        for input_tensor in self._input_tensors:
            copied_input_nodes = [node_mapping[node] for node
                                  in input_tensor.nodes]
            copied_input_tensor = Tensor(copied_input_nodes,
                                         input_tensor.shape)
            copied_input_tensors.append(copied_input_tensor)

        copied_output_tensors: List[Tensor] = list()
        for output_tensor in self._output_tensors:
            copied_output_nodes = [node_mapping[node] for node
                                   in output_tensor.nodes]
            copied_output_tensor = Tensor(copied_output_nodes,
                                          output_tensor.shape)
            copied_output_tensors.append(copied_output_tensor)

        copied_computational_graph = ComputationalGraph(
            copied_input_tensors, copied_output_tensors)
        copied_computational_graph.compile()

        return copied_computational_graph

    @property
    def input_tensors(self) -> List[Tensor]:
        return self._input_tensors

    @input_tensors.setter
    def input_tensors(self, value: List[Tensor]) -> None:
        self._input_tensors = value

    @property
    def output_tensors(self) -> List[Tensor]:
        return self._output_tensors

    @output_tensors.setter
    def output_tensors(self, value: List[Tensor]) -> None:
        self._output_tensors = value

    @property
    def input_nodes(self) -> List[ComputationalNode]:
        return self._input_nodes

    @property
    def output_nodes(self) -> List[ComputationalNode]:
        return self._output_nodes

    @property
    def evaluation_order(self) -> Tuple[ComputationalNode]:
        return self._evaluation_order
