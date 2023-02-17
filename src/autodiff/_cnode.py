from __future__ import annotations

from abc import ABC, abstractmethod
from math import atan2, cos, exp, log, nan, sin, tanh
from typing import Tuple


# TODO: rename to _cnode
class ComputationalNode(ABC):

    _operands: Tuple[ComputationalNode, ...]
    _result: float

    def __init__(
            self,
            operands: Tuple[ComputationalNode, ...],
            result: float
            ) -> None:
        self._operands = operands
        self._result = result

    @abstractmethod
    def evaluate(self, /) -> None:
        ...

    @abstractmethod
    def differentiate(self, i: int, /) -> ComputationalNode:
        ...

    @property
    def operands(self) -> Tuple[ComputationalNode, ...]:
        return self._operands

    @property
    def result(self) -> float:
        return self._result

    @result.setter
    def result(self, value: float) -> None:
        self._result = value


class EmptyComputationalNode(ComputationalNode):

    def __init__(self) -> None:
        super().__init__(operands=(), result=nan)

    def evaluate(self, /) -> None:
        raise NotImplementedError

    def differentiate(self, i: int, /) -> ComputationalNode:
        raise NotImplementedError


class Variable(ComputationalNode):

    def __init__(self, value: float = nan) -> None:
        super().__init__(operands=(), result=value)

    def evaluate(self, /) -> None:
        pass

    def differentiate(self, i: int, /) -> ComputationalNode:
        raise NotImplementedError


class Operation(ComputationalNode):

    def __init__(self, *operands: ComputationalNode) -> None:
        super().__init__(operands=operands, result=nan)


class Addition(Operation):

    def evaluate(self, /) -> None:
        self._result = self._operands[0].result + self._operands[1].result

    def differentiate(self, i: int, /) -> ComputationalNode:
        match i:
            case 0:
                return Variable(value=1.0)
            case 1:
                return Variable(value=1.0)
            case _:
                raise NotImplementedError


class Subtraction(Operation):

    def evaluate(self, /) -> None:
        self._result = self._operands[0].result - self._operands[1].result

    def differentiate(self, i: int, /) -> ComputationalNode:
        match i:
            case 0:
                return Variable(value=1.0)
            case 1:
                return Variable(value=-1.0)
            case _:
                raise NotImplementedError


class Multiplication(Operation):

    def evaluate(self, /) -> None:
        self._result = self._operands[0].result * self._operands[1].result

    def differentiate(self, i: int, /) -> ComputationalNode:
        match i:
            case 0:
                return self._operands[1]
            case 1:
                return self._operands[0]
            case _:
                raise NotImplementedError


class Division(Operation):

    def evaluate(self, /) -> None:
        self._result = self._operands[0].result / self._operands[1].result

    def differentiate(self, i: int, /) -> ComputationalNode:
        match i:
            case 0:
                return Division(
                    Variable(value=1.0),
                    self._operands[1])
            case 1:
                return Negation(
                    Division(
                        self._operands[0],
                        Exponentiation(
                            self._operands[1],
                            Variable(value=2.0))))
            case _:
                raise NotImplementedError


class Negation(Operation):

    def evaluate(self, /) -> None:
        self._result = -self._operands[0].result

    def differentiate(self, i: int, /) -> ComputationalNode:
        match i:
            case 0:
                return Variable(value=-1.0)
            case _:
                raise NotImplementedError


class Exponentiation(Operation):

    def evaluate(self, /) -> None:
        self._result = self._operands[0].result ** self._operands[1].result

    def differentiate(self, i: int, /) -> ComputationalNode:
        match i:
            case 0:
                return Multiplication(
                    self._operands[1],
                    Exponentiation(
                        self._operands[0],
                        Subtraction(
                            self._operands[1],
                            Variable(value=1.0))))
            case _:
                raise NotImplementedError


class Arctangent2(Operation):

    def evaluate(self, /) -> None:
        self._result = atan2(self._operands[0].result,
                             self._operands[1].result)

    def differentiate(self, i: int, /) -> ComputationalNode:
        match i:
            case 0:
                return Division(
                    self._operands[1],
                    Addition(
                        Exponentiation(
                            self._operands[0],
                            Variable(value=2.0)),
                        Exponentiation(
                            self._operands[1],
                            Variable(value=2.0))))
            case 1:
                return Division(
                    Negation(
                        self._operands[0]),
                    Addition(
                        Exponentiation(
                            self._operands[0],
                            Variable(value=2.0)),
                        Exponentiation(
                            self._operands[1],
                            Variable(value=2.0))))
            case _:
                raise NotImplementedError


class Cosine(Operation):

    def evaluate(self, /) -> None:
        self._result = cos(self._operands[0].result)

    def differentiate(self, i: int, /) -> ComputationalNode:
        match i:
            case 0:
                return Negation(Sine(self._operands[0]))
            case _:
                raise NotImplementedError


class HyperbolicTangent(Operation):

    def evaluate(self, /) -> None:
        self._result = tanh(self._operands[0].result)

    def differentiate(self, i: int, /) -> ComputationalNode:
        match i:
            case 0:
                return Subtraction(
                    Variable(value=1.0),
                    Exponentiation(
                        self,
                        Variable(value=2.0)))
            case _:
                raise NotImplementedError


class Sine(Operation):

    def evaluate(self, /) -> None:
        self._result = sin(self._operands[0].result)

    def differentiate(self, i: int, /) -> ComputationalNode:
        match i:
            case 0:
                return Cosine(self._operands[0])
            case _:
                raise NotImplementedError


class Exponential(Operation):

    def evaluate(self, /) -> None:
        self._result = exp(self._operands[0].result)

    def differentiate(self, i: int, /) -> ComputationalNode:
        match i:
            case 0:
                return self
            case _:
                raise NotImplementedError


class Logarithm(Operation):

    def evaluate(self, /) -> None:
        self._result = log(self._operands[0].result)

    def differentiate(self, i: int, /) -> ComputationalNode:
        match i:
            case 0:
                return Division(
                    Variable(value=1.0),
                    self._operands[0])
            case _:
                raise NotImplementedError
