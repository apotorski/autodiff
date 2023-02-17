from copy import copy
from typing import Dict, List, Optional

from ._cnode import Addition, ComputationalNode, Multiplication, Variable
from .cgraph import ComputationalGraph
from .tensor_api import Tensor


def _differentiate_through_forward_accumulation(
        computational_graph: ComputationalGraph,
        differentiated_operand_nodes: List[ComputationalNode],
        differentiated_result_nodes: List[ComputationalNode],
        seed_indice: int = 0
        ) -> List[ComputationalNode]:
    gradient_node_mapping: Dict[ComputationalNode, ComputationalNode] = dict()

    gradient_node_mapping[differentiated_operand_nodes[seed_indice]] = \
        Variable(value=1.0)

    for node in computational_graph.evaluation_order:
        operand_nodes_with_gradient = tuple(
            (i, node) for i, node in enumerate(node.operands)
            if node in gradient_node_mapping)

        if operand_nodes_with_gradient:
            gradient_node_mapping[node] = Variable(value=0.0)

            for i, operand_node in operand_nodes_with_gradient:
                gradient_node_mapping[node] = \
                    Addition(
                        gradient_node_mapping[node],
                        Multiplication(
                            gradient_node_mapping[operand_node],
                            node.differentiate(i)))

    return [gradient_node_mapping.get(node, Variable(value=0.0))
            for node in differentiated_result_nodes]


def _differentiate_through_reverse_accumulation(
        computational_graph: ComputationalGraph,
        differentiated_operand_nodes: List[ComputationalNode],
        differentiated_result_nodes: List[ComputationalNode],
        seed_indice: int = 0
        ) -> List[ComputationalNode]:
    gradient_node_mapping: Dict[ComputationalNode, ComputationalNode] = dict()
    nodes_requiring_gradient = set(differentiated_operand_nodes)

    for node in computational_graph.evaluation_order:
        if any(node in nodes_requiring_gradient for node in node.operands):
            nodes_requiring_gradient.add(node)

    gradient_node_mapping[differentiated_result_nodes[seed_indice]] = \
        Variable(value=1.0)

    for node in reversed(computational_graph.evaluation_order):
        if node in gradient_node_mapping:

            for i, operand_node in enumerate(node.operands):
                if operand_node in nodes_requiring_gradient:

                    if operand_node not in gradient_node_mapping:
                        gradient_node_mapping[operand_node] = \
                            Variable(value=0.0)

                    gradient_node_mapping[operand_node] = \
                        Addition(
                            gradient_node_mapping[operand_node],
                            Multiplication(
                                node.differentiate(i),
                                gradient_node_mapping[node]))

    return [gradient_node_mapping.get(node, Variable(value=0.0))
            for node in differentiated_operand_nodes]


def differentiate(
        computational_graph: ComputationalGraph,
        differentiated_operands: Optional[Tensor] = None,
        differentiated_results: Optional[Tensor] = None
        ) -> Tensor:
    """Differentiate computational graph according to chain rule and return
    matrix of partial derivatives.
    """
    if differentiated_operands is None:
        differentiated_operand_nodes = computational_graph.input_nodes
    else:
        differentiated_operand_nodes = differentiated_operands.nodes

    if differentiated_results is None:
        differentiated_result_nodes = computational_graph.output_nodes
    else:
        differentiated_result_nodes = differentiated_results.nodes

    derivative_tensor_shape = (
        len(differentiated_result_nodes),
        len(differentiated_operand_nodes)
    )

    if derivative_tensor_shape[0] > derivative_tensor_shape[1]:
        transposed_derivative_nodes: List[ComputationalNode] = list()

        for seed_indice in range(derivative_tensor_shape[1]):
            transposed_derivative_nodes += \
                _differentiate_through_forward_accumulation(
                    computational_graph, differentiated_operand_nodes,
                    differentiated_result_nodes, seed_indice)

        transposed_derivative_tensor_shape = \
            tuple(reversed(derivative_tensor_shape))
        derivative_tensor = Tensor(
            transposed_derivative_nodes, transposed_derivative_tensor_shape) \
                .transpose()
    else:
        derivative_nodes: List[ComputationalNode] = list()

        for seed_indice in range(derivative_tensor_shape[0]):
            derivative_nodes += \
                _differentiate_through_reverse_accumulation(
                    computational_graph, differentiated_operand_nodes,
                    differentiated_result_nodes, seed_indice)

        derivative_tensor = Tensor(
            derivative_nodes, derivative_tensor_shape)

    return derivative_tensor


def derive_gradient(
        computational_graph: ComputationalGraph
        ) -> ComputationalGraph:
    if len(computational_graph.output_nodes) != 1:
        raise ValueError('Scalar-valued function is required!')

    gradient_computational_graph = copy(computational_graph)

    gradient_output_interface = differentiate(gradient_computational_graph) \
        .transpose()
    gradient_computational_graph.output_tensors = [gradient_output_interface]
    gradient_computational_graph.compile()

    return gradient_computational_graph


def derive_jacobian(
        computational_graph: ComputationalGraph
        ) -> ComputationalGraph:
    jacobian_computational_graph = copy(computational_graph)

    jacobian_output_interface = differentiate(jacobian_computational_graph)
    jacobian_computational_graph.output_tensors = [jacobian_output_interface]
    jacobian_computational_graph.compile()

    return jacobian_computational_graph


def derive_hessian(
        computational_graph: ComputationalGraph
        ) -> ComputationalGraph:
    return derive_jacobian(derive_gradient(computational_graph))
