from __future__ import annotations

import ast
import operator
import re
from fractions import Fraction
from typing import Any

import sympy as sp

from umbac.base_converter import BaseConverter
from umbac.errors import BaseConversionError, DomainError, ParseError


DEFAULT_CONSTANTS = {"pi": sp.pi, "e": sp.E, "E": sp.E, "I": sp.I, "oo": sp.oo, "inf": sp.oo}
DEFAULT_FUNCTIONS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
    "Abs": sp.Abs, "abs": sp.Abs, "sign": sp.sign,
    "floor": sp.floor, "ceiling": sp.ceiling,
    "factorial": sp.factorial, "gamma": sp.gamma, "erf": sp.erf,
    "diff": sp.diff, "integrate": sp.integrate, "limit": sp.limit,
    "summation": sp.summation, "product": sp.product,
    "simplify": sp.simplify, "expand": sp.expand, "factor": sp.factor,
    "solve": sp.solve, "Matrix": sp.Matrix, "Function": sp.Function,
    "Derivative": sp.Derivative, "Eq": sp.Eq, "det": sp.det, "trace": sp.trace,
}
BLOCKED_FUNCTIONS = {"open", "eval", "exec", "compile", "input", "getattr", "setattr", "delattr"}


class SafeExpressionParser:
    """Parse an intentionally small arithmetic AST without evaluating Python code."""

    def __init__(self, constants: dict[str, Any] | None = None, functions: dict[str, Any] | None = None,
                 blocked_names: set[str] | None = None) -> None:
        self.constants = {**DEFAULT_CONSTANTS, **(constants or {})}
        self.functions = {**DEFAULT_FUNCTIONS, **(functions or {})}
        self.blocked_names = blocked_names or set()
        self.binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                       ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
        self.unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def parse(self, expression: str, default_base: int = 10) -> Any:
        BaseConverter.validate_base(default_base)
        text = expression.strip().replace("^", "**")
        text = re.sub(r"\bln\s*\(", "log(", text)
        text = re.sub(r"\[([^\]]+)\]_(\d+)", lambda match: f"base({int(match.group(2))}, {match.group(1)!r})", text)
        string_literals: list[str] = []
        number_literals: list[Fraction] = []

        def protect_string(match: re.Match[str]) -> str:
            string_literals.append(ast.literal_eval(match.group(0)))
            return f"__umbac_literal_{len(string_literals) - 1}__"

        text = re.sub(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"", protect_string, text)

        def protect_number(match: re.Match[str]) -> str:
            token = match.group(0)
            try:
                number_literals.append(BaseConverter.parse_fraction(token, default_base))
            except BaseConversionError:
                return token
            return f"__umbac_number_{len(number_literals) - 1}__"

        text = re.sub(r"(?<![A-Za-z0-9_])(?:[0-9][A-Za-z0-9]*(?:\.[A-Za-z0-9]*)?|\.[A-Za-z0-9]+)", protect_number, text)
        text = re.sub(r"(__umbac_number_\d+__)(?=[A-Za-z_(])", r"\1*", text)
        text = re.sub(r"(?<![A-Za-z0-9_])([0-9]+)(?=[A-Za-z_(])", r"\1*", text)
        if not text:
            raise ParseError("Введите выражение.")
        try:
            return self._visit(ast.parse(text, mode="eval").body, string_literals, number_literals, default_base)
        except ParseError:
            raise
        except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError) as error:
            raise ParseError(f"Не удалось разобрать выражение: {error}", "Используйте арифметику, разрешённые функции и явную обёртку base().") from error

    def _visit(self, node: ast.AST, string_literals: list[str], number_literals: list[Fraction], default_base: int) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            rational = BaseConverter.parse_fraction(str(node.value), default_base)
            return sp.Rational(rational.numerator, rational.denominator)
        if isinstance(node, ast.Name):
            if node.id in self.blocked_names:
                raise DomainError(f"Имя {node.id!r} запрещено текущей конфигурацией.")
            if node.id.startswith("__umbac_literal_"):
                index = int(node.id.removeprefix("__umbac_literal_").removesuffix("__"))
                return string_literals[index]
            if node.id.startswith("__umbac_number_"):
                index = int(node.id.removeprefix("__umbac_number_").removesuffix("__"))
                rational = number_literals[index]
                return sp.Rational(rational.numerator, rational.denominator)
            if default_base <= 36 and node.id.isascii() and node.id.isalpha():
                try:
                    rational = BaseConverter.parse_fraction(node.id, default_base)
                    return sp.Rational(rational.numerator, rational.denominator)
                except BaseConversionError:
                    pass
            if node.id in self.constants:
                return self.constants[node.id]
            return sp.Symbol(node.id)
        if isinstance(node, ast.BinOp) and type(node.op) in self.binary:
            left = self._visit(node.left, string_literals, number_literals, default_base)
            right = self._visit(node.right, string_literals, number_literals, default_base)
            if isinstance(node.op, ast.Div) and right == 0:
                raise DomainError("Деление на ноль.", "Укажите ненулевой знаменатель.")
            if isinstance(node.op, ast.Pow) and isinstance(right, sp.Integer) and abs(int(right)) > 1000:
                raise ParseError("Слишком большая степень.", "Уменьшите показатель степени.")
            return self.binary[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in self.unary:
            return self.unary[type(node.op)](self._visit(node.operand, string_literals, number_literals, default_base))
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._visit(item, string_literals, number_literals, default_base) for item in node.elts]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.keywords:
                raise ParseError("Именованные аргументы не поддерживаются.")
            if node.func.id == "base":
                if len(node.args) != 2:
                    raise ParseError("Формат литерала: base(16, '1F.8').")
                literal = node.args[1]
                if isinstance(literal, ast.Name) and literal.id.startswith("__umbac_literal_"):
                    index = int(literal.id.removeprefix("__umbac_literal_").removesuffix("__"))
                    digits = string_literals[index]
                elif isinstance(literal, ast.Constant) and isinstance(literal.value, str):
                    digits = literal.value
                else:
                    raise ParseError("Формат литерала: base(16, '1F.8').")
                base = self._visit(node.args[0], string_literals, number_literals, 10)
                if not isinstance(base, sp.Integer):
                    raise ParseError("Основание в base() должно быть целым числом.")
                rational = BaseConverter.parse_fraction(digits, int(base))
                return sp.Rational(rational.numerator, rational.denominator)
            function = self.functions.get(node.func.id)
            arguments = [self._visit(argument, string_literals, number_literals, default_base) for argument in node.args]
            if function is None:
                if node.func.id in BLOCKED_FUNCTIONS or node.func.id.startswith("_") or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", node.func.id):
                    raise ParseError(f"Функция {node.func.id!r} не разрешена.")
                function = sp.Function(node.func.id)
            if node.func.id == "log" and arguments and arguments[0] == 0:
                raise DomainError("Логарифм нуля не определён.")
            return function(*arguments)
        raise ParseError("Выражение содержит запрещённую или неподдерживаемую конструкцию.")
