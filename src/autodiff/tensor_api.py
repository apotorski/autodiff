from __future__ import annotations

from functools import singledispatchmethod
from math import prod
from typing import List, Tuple, Type

import numpy as np

from ._cnode import (Addition, Arctangent2, ComputationalNode, Cosine,
                     Division, EmptyComputationalNode, Exponential,
                     Exponentiation, HyperbolicTangent, Logarithm,
                     Multiplication, Negation, Operation, Sine, Subtraction,
                     Variable)
from ._typing import NDArrayFloat


class Tensor:

    """Provides support for broadcasting and overloads operators."""

    _nodes: List[ComputationalNode]
    _shape: Tuple[int, ...]

    def __init__(
            self,
            nodes: List[ComputationalNode],
            shape: Tuple[int, ...]
            ) -> None:
        if len(nodes) != prod(shape):
            raise ValueError(f'Mismatch between number of provided nodes '
                             f'({len(nodes)}) and desired shape ({shape})!')

        self._nodes = nodes
        self._shape = shape

    @classmethod
    def from_tensors(
            cls,
            tensors: List[Tensor],
            shape: Tuple[int, ...]
            ) -> Tensor:
        nodes: List[ComputationalNode] = list()
        for tensor in tensors:
            nodes += tensor.nodes
        return cls(nodes=nodes, shape=shape)

    @singledispatchmethod
    @classmethod
    def as_tensor(cls, value: float | NDArrayFloat, /) -> Tensor:
        raise NotImplementedError(f'{type(value)} cannot be converted to '
                                  f'tensor!')

    @as_tensor.register(float)
    @classmethod
    def _(cls, value: float, /) -> Tensor:
        return cls(nodes=[Variable(value)], shape=())

    @as_tensor.register(np.ndarray)
    @classmethod
    def _(cls, value: NDArrayFloat, /) -> Tensor:
        return cls(nodes=[Variable(float(element)) for element in value.flat],
            shape=value.shape)

    @singledispatchmethod
    def __getitem__(
            self,
            key: int | Tuple[int, ...],
            /
            ) -> Tensor:
        raise TypeError(f'{type(key)} cannot be used as indice!')

    @__getitem__.register(int)
    def _(self, key: int, /) -> Tensor:
        if key < 0 or key > len(self):
            raise IndexError(f'Indice {key} is out of bounds for tensor with '
                             f'size {len(self)}!')

        return Tensor(nodes=[self._nodes[key]], shape=())

    @__getitem__.register(tuple)
    def _(self, key: Tuple[int, ...], /) -> Tensor:
        if len(self._shape) != len(key):
            raise IndexError(f'Indices {key} are incompatible with shape '
                             f'{self._shape}!')

        for axis, (indice, axis_size) in enumerate(zip(key, self._shape)):
            if indice < 0 or indice > axis_size:
                raise IndexError(f'Indice {key} is out of bounds for axis '
                                 f'{axis} with size {axis_size}!')

        node = self._nodes[Tensor.compute_indice(key, self._shape)]
        return Tensor(nodes=[node], shape=())

    @singledispatchmethod
    def __setitem__(
            self,
            key: int | Tuple[int, ...],
            value: Tensor,
            /
            ) -> None:
        raise TypeError(f'{type(key)} cannot be used as indice!')

    @__setitem__.register(int)
    def _(self, key: int, value: Tensor, /) -> None:
        if key < 0 or key > len(self):
            raise IndexError(f'Indice {key} is out of bounds for tensor with '
                             f'size {len(self)}!')

        self._nodes[key] = value.nodes[0]

    @__setitem__.register(tuple)
    def _(self, key: Tuple[int, ...], value: Tensor, /) -> None:
        if len(self._shape) != len(key):
            raise IndexError(f'Indices {key} are incompatible with shape '
                             f'{self._shape}!')

        for axis, (indice, axis_size) in enumerate(zip(key, self._shape)):
            if indice < 0 or indice > axis_size:
                raise IndexError(f'Indice {key} is out of bounds for axis '
                                 f'{axis} with size {axis_size}!')

        self._nodes[Tensor.compute_indice(key, self._shape)] = value.nodes[0]

    @staticmethod
    def normalize_operand(operand: Tensor | float | NDArrayFloat) -> Tensor:
        if isinstance(operand, Tensor):
            return operand
        else:
            return Tensor.as_tensor(operand)

    @staticmethod
    def broadcast_shapes(
            shape_1: Tuple[int, ...],
            shape_2: Tuple[int, ...]
            ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        if len(shape_1) > len(shape_2):
            return shape_1, (1,)*(len(shape_1) - len(shape_2)) + shape_2
        else:
            return (1,)*(len(shape_2) - len(shape_1)) + shape_1, shape_2

    @staticmethod
    def compute_indice(
            indices: Tuple[int, ...],
            shape: Tuple[int, ...]
            ) -> int:
        return sum(indice*prod(shape[axis + 1:])
            for axis, indice in enumerate(indices))

    @staticmethod
    def compute_indices(
            indice: int,
            shape: Tuple[int, ...]
            ) -> Tuple[int, ...]:
        return tuple(indice // prod(shape[axis + 1:]) % shape[axis]
            for axis in range(len(shape)))

    @classmethod
    def apply_binary_operation(
            cls,
            operand_1: Tensor | float | NDArrayFloat,
            operand_2: Tensor | float | NDArrayFloat,
            operation: Type[Operation],
            /
            ) -> Tensor:
        _operand_1 = Tensor.normalize_operand(operand_1)
        _operand_2 = Tensor.normalize_operand(operand_2)

        broadcasted_operand_1_shape, broadcasted_operand_2_shape = \
            Tensor.broadcast_shapes(_operand_1.shape, _operand_2.shape)

        for operand_1_axis_size, operand_2_axis_size \
            in zip(broadcasted_operand_1_shape, broadcasted_operand_2_shape):
                if not (operand_1_axis_size == operand_2_axis_size
                    or operand_1_axis_size == 1 or operand_2_axis_size == 1):
                        raise ValueError(f'{_operand_1.shape} and '
                                         f'{_operand_2.shape} cannot be '
                                         f'broadcasted together!')

        result_shape = tuple(
            max(operand_1_axis_size, operand_2_axis_size)
            for operand_1_axis_size, operand_2_axis_size
            in zip(broadcasted_operand_1_shape, broadcasted_operand_2_shape))

        result_nodes: List[ComputationalNode] = [EmptyComputationalNode()] \
            * prod(result_shape)
        for result_indice in range(prod(result_shape)):
            result_indices = Tensor.compute_indices(
                result_indice, result_shape)

            broadcasted_operand_1_indices = tuple(
                0 if axis_size == 1 else result_indices[axis]
                for axis, axis_size in enumerate(broadcasted_operand_1_shape))
            broadcasted_operand_2_indices = tuple(
                0 if axis_size == 1 else result_indices[axis]
                for axis, axis_size in enumerate(broadcasted_operand_2_shape))

            broadcasted_operand_1_indice = Tensor.compute_indice(
                broadcasted_operand_1_indices, broadcasted_operand_1_shape)
            broadcasted_operand_2_indice = Tensor.compute_indice(
                broadcasted_operand_2_indices, broadcasted_operand_2_shape)

            result_nodes[result_indice] = operation(
                _operand_1.nodes[broadcasted_operand_1_indice],
                _operand_2.nodes[broadcasted_operand_2_indice])

        result = cls(nodes=result_nodes, shape=result_shape)

        return result

    @classmethod
    def multiply_matrices(
            cls,
            operand_1: Tensor | float | NDArrayFloat,
            operand_2: Tensor | float | NDArrayFloat,
            /
            ) -> Tensor:
        _operand_1 = Tensor.normalize_operand(operand_1)
        _operand_2 = Tensor.normalize_operand(operand_2)

        broadcasted_operand_1_shape, broadcasted_operand_2_shape = \
            Tensor.broadcast_shapes(_operand_1.shape, _operand_2.shape)

        shared_operand_1_shape = broadcasted_operand_1_shape[:-2]
        shared_operand_2_shape = broadcasted_operand_2_shape[:-2]

        for operand_1_axis_size, operand_2_axis_size \
            in zip(shared_operand_1_shape, shared_operand_2_shape):
                if not (operand_1_axis_size == operand_2_axis_size
                    or operand_1_axis_size == 1 or operand_2_axis_size == 1):
                        raise ValueError(f'{_operand_1.shape} and '
                                         f'{_operand_2.shape} cannot be '
                                         f'broadcasted together!')

        if _operand_1.shape[-1] != _operand_2.shape[-2]:
            raise ValueError(f'Matrices with shapes {_operand_1.shape} and '
                             f'{_operand_2.shape} cannot be multiplied!')

        m, n = _operand_1.shape[-2:]
        _, p = _operand_2.shape[-2:]

        result_shape = tuple(
            max(operand_1_axis_size, operand_2_axis_size)
            for operand_1_axis_size, operand_2_axis_size
            in zip(shared_operand_1_shape, shared_operand_2_shape)) \
                + (m, p)

        full_shape = result_shape + (n,)

        result_nodes: List[ComputationalNode] = [Variable(value=0.0)] \
            * prod(result_shape)
        for indice in range(prod(full_shape)):
            indices = Tensor.compute_indices(indice, full_shape)

            shared_indices, (i, j, k) = indices[:-3], indices[-3:]

            broadcasted_operand_1_indices = tuple(
                0 if axis_size == 1 else shared_indices[axis]
                for axis, axis_size in enumerate(shared_operand_1_shape)) \
                    + (i, k)
            broadcasted_operand_2_indices = tuple(
                0 if axis_size == 1 else shared_indices[axis]
                for axis, axis_size in enumerate(shared_operand_2_shape)) \
                    + (k, j)

            broadcasted_operand_1_indice = Tensor.compute_indice(
                broadcasted_operand_1_indices, broadcasted_operand_1_shape)
            broadcasted_operand_2_indice = Tensor.compute_indice(
                broadcasted_operand_2_indices, broadcasted_operand_2_shape)

            result_indice = Tensor.compute_indice(
                shared_indices + (i, j), result_shape)

            result_nodes[result_indice] = \
                Addition(
                    result_nodes[result_indice],
                    Multiplication(
                        _operand_1.nodes[broadcasted_operand_1_indice],
                        _operand_2.nodes[broadcasted_operand_2_indice]))

        result = cls(nodes=result_nodes, shape=result_shape)

        return result

    @classmethod
    def apply_unary_operation(
            cls,
            operand: Tensor,
            operation: Type[Operation],
            /
            ) -> Tensor:
        result_nodes: List[ComputationalNode] = \
            [operation(operand_node) for operand_node in operand.nodes]
        result_shape = operand.shape
        result = cls(nodes=result_nodes, shape=result_shape)

        return result

    def __add__(self, other: Tensor | float | NDArrayFloat, /) -> Tensor:
        return Tensor.apply_binary_operation(self, other, Addition)

    def __sub__(self, other: Tensor | float | NDArrayFloat, /) -> Tensor:
        return Tensor.apply_binary_operation(self, other, Subtraction)

    def __mul__(self, other: Tensor | float | NDArrayFloat, /) -> Tensor:
        return Tensor.apply_binary_operation(self, other, Multiplication)

    def __matmul__(self, other: Tensor | float | NDArrayFloat, /) -> Tensor:
        return Tensor.multiply_matrices(self, other)

    def __truediv__(self, other: Tensor | float | NDArrayFloat, /) -> Tensor:
        return Tensor.apply_binary_operation(self, other, Division)

    def __pow__(self, other: Tensor | float | NDArrayFloat, /) -> Tensor:
        return Tensor.apply_binary_operation(self, other, Exponentiation)

    def __radd__(self, other: Tensor | float | NDArrayFloat, /) -> Tensor:
        return Tensor.apply_binary_operation(other, self, Addition)

    def __rsub__(self, other: Tensor | float | NDArrayFloat, /) -> Tensor:
        return Tensor.apply_binary_operation(other, self, Subtraction)

    def __rmul__(self, other: Tensor | float | NDArrayFloat, /) -> Tensor:
        return Tensor.apply_binary_operation(other, self, Multiplication)

    def __rmatmul__(self, other: Tensor | float | NDArrayFloat, /) -> Tensor:
        return Tensor.multiply_matrices(other, self)

    def __rtruediv__(self, other: Tensor | float | NDArrayFloat, /) -> Tensor:
        return Tensor.apply_binary_operation(other, self, Division)

    def __rpow__(self, other: Tensor | float | NDArrayFloat, /) -> Tensor:
        return Tensor.apply_binary_operation(other, self, Exponentiation)

    def __pos__(self, /) -> Tensor:
        return self

    def __neg__(self, /) -> Tensor:
        return Tensor.apply_unary_operation(self, Negation)

    def __len__(self, /) -> int:
        return len(self._nodes)

    def __repr__(self, /) -> str:
        return (f'{self.__class__.__name__}('
                f'nodes={[node.result for node in self._nodes]}, '
                f'shape={self._shape})')

    def reshape(self, shape: Tuple[int, ...], /) -> Tensor:
        if prod(self._shape) != prod(shape):
            raise ValueError(f'{self._shape} cannot be reshaped into {shape}!')

        self._shape = shape

        return self

    def sum(self, /) -> Tensor:
        result = Variable(value=0.0)
        for node in self._nodes:
            result = Addition(result, node)
        return Tensor(nodes=[result], shape=())

    def mean(self, /) -> Tensor:
        return self.sum() / float(len(self))

    def transpose(self, /) -> Tensor:
        result_nodes: List[ComputationalNode] = list()
        result_shape = tuple(reversed(self._shape))

        for transposed_indice in range(prod(result_shape)):
            transposed_indices = Tensor.compute_indices(
                transposed_indice, result_shape)
            indices = tuple(reversed(transposed_indices))
            indice = Tensor.compute_indice(indices, self._shape)
            result_node = self._nodes[indice]
            result_nodes.append(result_node)

        result = Tensor(nodes=result_nodes, shape=result_shape)

        return result

    def assign(self, value: NDArrayFloat, /) -> None:
        if len(self) != value.size:
            raise ValueError(f'Size of assigned value ({value.size}) and size '
                             f'of tensor ({len(self)}) are different!')

        for node, element in zip(self._nodes, value.flat):
            node.result = float(element)

    def access(self, /) -> NDArrayFloat:
        return np.array([node.result for node in self._nodes]) \
            .reshape(self._shape)

    @property
    def nodes(self) -> List[ComputationalNode]:
        return self._nodes

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape


def parameter(*shape: int) -> Tensor:
    return Tensor([Variable() for _ in range(prod(shape))], shape)

def constant(value: float | NDArrayFloat, /) -> Tensor:
    return Tensor.as_tensor(value)

def empty(*shape: int) -> Tensor:
    return Tensor([EmptyComputationalNode()] * prod(shape), shape)


def atan2(y: Tensor, x: Tensor, /) -> Tensor:
    return Tensor.apply_binary_operation(y, x, Arctangent2)

def cos(x: Tensor, /) -> Tensor:
    return Tensor.apply_unary_operation(x, Cosine)

def exp(x: Tensor, /) -> Tensor:
    return Tensor.apply_unary_operation(x, Exponential)

def log(x: Tensor, /) -> Tensor:
    return Tensor.apply_unary_operation(x, Logarithm)

def sin(x: Tensor, /) -> Tensor:
    return Tensor.apply_unary_operation(x, Sine)

def tanh(x: Tensor, /) -> Tensor:
    return Tensor.apply_unary_operation(x, HyperbolicTangent)
