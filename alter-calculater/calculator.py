from __future__ import annotations

import sympy as sp
from umbac.base_converter import ALPHABET, MAX_BASE, BaseConverter
from umbac.expr_parser import SafeExpressionParser
from umbac.errors import BaseConversionError


DIGITS = ALPHABET
SUPPORTED_BASES = tuple(range(2, MAX_BASE + 1))
MAX_FRACTION_DIGITS = 48
PHYSICAL_CONSTANTS = {"c": "299792458", "G": "6.67430e-11", "h": "6.62607015e-34"}


def convert_base(value: str, source_base: int, target_base: int) -> dict[str, object]:
    """Convert a positional number, preserving its exact rational value internally."""
    converter = BaseConverter()
    try:
        number = converter.parse_fraction(value, source_base)
    except BaseConversionError as error:
        raise ValueError(str(error)) from error
    converted = converter.from_decimal(number, target_base, MAX_FRACTION_DIGITS)
    periodic = converter.from_decimal_periodic(number, target_base)
    exact = "(" not in periodic and not periodic.endswith("…")
    return {"value": converted + ("…" if not exact else ""), "exact": exact,
            "fraction": f"{number.numerator}/{number.denominator}",
            "digits": "0-9, A-Z, a-z, +, / (значения 0-63)"}


def evaluate(expression: str) -> dict[str, str | None]:
    functions = {
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "asin": sp.asin,
        "acos": sp.acos, "atan": sp.atan, "sinh": sp.sinh, "cosh": sp.cosh,
        "tanh": sp.tanh, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
        "Abs": sp.Abs, "abs": sp.Abs, "sign": sp.sign, "factorial": sp.factorial,
        "gamma": sp.gamma, "erf": sp.erf, "diff": sp.diff, "integrate": sp.integrate,
        "limit": sp.limit, "summation": sp.summation, "product": sp.product,
        "simplify": sp.simplify, "expand": sp.expand, "factor": sp.factor,
        "solve": sp.solve, "Matrix": sp.Matrix, "det": sp.det, "trace": sp.trace,
    }
    constants = {
        "pi": sp.pi, "e": sp.E, "E": sp.E, "I": sp.I, "oo": sp.oo,
        "c": sp.Integer(299792458), "G": sp.Rational("6.67430e-11"),
        "h": sp.Rational("6.62607015e-34"), "hbar": sp.Rational("1.054571817e-34"),
        "k_B": sp.Rational("1.380649e-23"), "e_charge": sp.Rational("1.602176634e-19"),
        "m_e": sp.Rational("9.1093837139e-31"), "m_p": sp.Rational("1.67262192595e-27"),
        "AU": sp.Integer(149597870700), "M_sun": sp.Rational("1.98847e30"),
        "R_sun": sp.Rational("6.957e8"), "M_earth": sp.Rational("5.9722e24"),
    }
    result = SafeExpressionParser(constants, functions).parse(expression)

    exact = sp.sstr(result)
    approximation = None
    if isinstance(result, sp.MatrixBase):
        approximation = str(result.evalf(10))
    elif isinstance(result, sp.Basic) and not result.free_symbols:
        try:
            approximation = str(result.evalf(14))
        except (TypeError, ValueError):
            approximation = None
    return {"exact": exact, "approximation": approximation}
