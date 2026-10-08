from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import mpmath as mp
import numpy as np
import sympy as sp

from umbac.analysis import AnalysisResult, MathAnalysisEngine
from umbac.astro import AstrophysicsEngine
from umbac.base_converter import BaseConverter
from umbac.config import CalcConfig
from umbac.constants import CONSTANTS, Constants
from umbac.errors import BaseConversionError, DomainError, SolverError, UMBACError
from umbac.expr_parser import SafeExpressionParser
from umbac.quantum import QuantumEngine


@dataclass(slots=True)
class CalcResult:
    """Serializable result envelope for all calculator operations."""

    ok: bool
    value_decimal: Any = None
    value_base_out: Any = None
    latex: str | None = None
    data: Any = None
    method: str | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, (sp.Basic, mp.mpf)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


class UniversalAdvancedCalculator:
    """Coordinated facade for base conversion, analysis, quantum physics, and astrophysics."""

    def __init__(self, config: CalcConfig | None = None, units: str = "atomic") -> None:
        self.config = config or CalcConfig()
        self.base_converter = BaseConverter(self.config)
        self.constants = Constants()
        constant_values = {name: sp.Rational(str(item.value)) for name, item in CONSTANTS.items()}
        angle_scale = sp.pi / 180 if self.config.angle_unit == "deg" else 1
        angle_result_scale = 180 / sp.pi if self.config.angle_unit == "deg" else 1
        self.functions: dict[str, Any] = {
            "sin": lambda value: sp.sin(value * angle_scale),
            "cos": lambda value: sp.cos(value * angle_scale),
            "tan": lambda value: sp.tan(value * angle_scale),
            "asin": lambda value: sp.asin(value) * angle_result_scale,
            "acos": lambda value: sp.acos(value) * angle_result_scale,
            "atan": lambda value: sp.atan(value) * angle_result_scale,
            "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
            "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
            "Abs": sp.Abs, "abs": sp.Abs, "sign": sp.sign,
            "floor": sp.floor, "ceiling": sp.ceiling,
            "factorial": sp.factorial, "gamma": sp.gamma, "erf": sp.erf,
            "diff": sp.diff, "integrate": sp.integrate, "limit": sp.limit,
            "summation": sp.summation, "product": sp.product,
            "simplify": sp.simplify, "expand": sp.expand, "factor": sp.factor,
            "solve": sp.solve, "Matrix": sp.Matrix, "det": sp.det, "trace": sp.trace,
        }
        self.parser = SafeExpressionParser(
            {"pi": sp.pi, "e": sp.E, "E": sp.E, "I": sp.I, "oo": sp.oo, "inf": sp.oo, "nan": sp.nan,
             **constant_values}, self.functions,
            blocked_names=set() if self.config.complex_enabled else {"I"},
        )
        self.analysis = MathAnalysisEngine(self.config, self.parser)
        self.quantum = QuantumEngine(units=units)
        self.astro = AstrophysicsEngine()
        self._operations: dict[str, Callable[..., Any]] = {
            "base": self._convert,
            "analysis.evaluate": self._evaluate,
            "analysis.derive": self._derive,
            "analysis.integrate": self._integrate,
            "analysis.limit": self._limit,
            "analysis.taylor": self._taylor,
            "analysis.fourier": self._fourier,
            "analysis.ode": self._ode,
            "analysis.simplify_check": self._simplify_check,
            "quantum.schrodinger": self._schrodinger,
            "quantum.eigen": self._eigen,
            "quantum.entropy": self._entropy,
            "quantum.entanglement": self._entanglement,
            "astro.schwarzschild": self._schwarzschild,
            "astro.friedmann": self._friedmann,
            "astro.chandrasekhar": self._chandrasekhar,
            "astro.lane_emden": self._lane_emden,
            "constants.get": self._constant,
        }

    @property
    def operations(self) -> tuple[str, ...]:
        """List operation identifiers supported by compute()."""
        return tuple(self._operations)

    def compute(self, expression: str = "", operation: str = "base", base_in: int = 10,
                base_out: int = 10, **params: Any) -> CalcResult:
        """Validate, dispatch, and normalize one calculation into a CalcResult."""
        try:
            self.base_converter.validate_base(base_in)
            self.base_converter.validate_base(base_out)
            handler = self._operations.get(operation)
            if handler is None:
                raise SolverError(f"Неизвестная операция: {operation}", f"Доступные операции: {', '.join(self.operations)}")
            result = handler(expression=expression, base_in=base_in, base_out=base_out, **params)
            if not self.config.complex_enabled and isinstance(result, AnalysisResult):
                if isinstance(result.symbolic, sp.Basic) and result.symbolic.has(sp.I):
                    raise DomainError("Комплексные результаты отключены в CalcConfig.")
            return self._wrap_result(result, base_out, params)
        except UMBACError as error:
            return CalcResult(False, error=str(error), hint=error.hint or None)
        except (ArithmeticError, TypeError, ValueError, KeyError, np.linalg.LinAlgError) as error:
            return CalcResult(False, error=str(error), hint="Проверьте параметры и область определения операции.")
        except Exception as error:
            return CalcResult(False, error=f"Ошибка вычислительного движка: {error}",
                              hint="Проверьте входные данные; неизвестная ошибка не скрывается как успешное решение.")

    def _convert(self, expression: str, base_in: int, base_out: int, **_: Any) -> dict[str, Any]:
        fraction = self.base_converter.parse_fraction(expression, base_in)
        return {"decimal": str(self.base_converter.to_decimal(expression, base_in)),
                "base_out": self.base_converter.from_decimal(fraction, base_out),
                "exact_fraction": f"{fraction.numerator}/{fraction.denominator}",
                "periodic": self.base_converter.from_decimal_periodic(fraction, base_out)}

    def _expr(self, expression: str, base_in: int = 10) -> Any:
        if not expression:
            raise ValueError("Для операции требуется expression.")
        return self.parser.parse(expression, default_base=base_in)

    def _evaluate(self, expression: str, base_in: int = 10, **_: Any) -> AnalysisResult:
        return self.analysis._result(self._expr(expression, base_in))

    def _derive(self, expression: str, base_in: int = 10, **params: Any) -> AnalysisResult:
        return self.analysis.derive(self._expr(expression, base_in), params.get("var", "x"), int(params.get("order", 1)), params.get("point"))

    def _integrate(self, expression: str, base_in: int = 10, **params: Any) -> AnalysisResult:
        return self.analysis.integrate(self._expr(expression, base_in), params.get("var", "x"), params.get("limits"))

    def _limit(self, expression: str, base_in: int = 10, **params: Any) -> AnalysisResult:
        return self.analysis.find_limit(self._expr(expression, base_in), params.get("var", "x"), params.get("point", 0), params.get("direction", "+-"))

    def _taylor(self, expression: str, base_in: int = 10, **params: Any) -> AnalysisResult:
        return self.analysis.taylor(self._expr(expression, base_in), params.get("var", "x"), params.get("point", 0), int(params.get("order", 5)))

    def _fourier(self, expression: str, base_in: int = 10, **params: Any) -> AnalysisResult:
        return self.analysis.fourier_series(self._expr(expression, base_in), params.get("var", "x"), params.get("period", 2 * sp.pi), int(params.get("n_terms", 10)))

    def _ode(self, expression: str, base_in: int = 10, **params: Any) -> AnalysisResult:
        return self.analysis.solve_ode(self._expr(expression, base_in), params.get("func", "y"), params.get("ics"), params.get("order"))

    def _simplify_check(self, expression: str, base_in: int = 10, **params: Any) -> AnalysisResult:
        return self.analysis.simplify_check(self._expr(expression, base_in), self._expr(params["other"], base_in))

    def _schrodinger(self, expression: str, **params: Any) -> dict[str, Any]:
        return self.quantum.schrodinger_solver(expression or params.get("potential", "infinite_well"),
                                               float(params.get("x_min", -5)), float(params.get("x_max", 5)),
                                               int(params.get("n_points", 600)), int(params.get("n_states", 6)),
                                               float(params["mass"]) if "mass" in params else None)

    def _eigen(self, expression: str, base_in: int = 10, **_: Any) -> dict[str, Any]:
        matrix = self._expr(expression, base_in)
        return self.quantum.matrix_eigenstates(np.asarray(matrix, dtype=np.complex128))

    def _entropy(self, expression: str, base_in: int = 10, **_: Any) -> dict[str, Any]:
        matrix = self._expr(expression, base_in)
        return {"von_neumann_entropy_bits": self.quantum.quantum_entropy(np.asarray(matrix, dtype=np.complex128))}

    def _entanglement(self, expression: str, **params: Any) -> dict[str, Any]:
        state = self.quantum.bell_state(params.get("kind", expression or "phi_plus"))
        dims = params.get("dims", (2, 2))
        rho = state["density_matrix"]
        reduced = self.quantum.partial_trace(rho, dims, params.get("keep", (0,)))
        return {"kind": state["kind"], "state": state["state"], "density_matrix": rho,
                "reduced_density_matrix": reduced, "reduced_entropy_bits": self.quantum.quantum_entropy(reduced),
                **self.quantum.peres_horodecki(rho, dims)}

    def _schwarzschild(self, expression: str, base_in: int = 10, **params: Any) -> dict[str, Any]:
        mass = params["mass"] if "mass" in params else self.parser.parse(expression, default_base=base_in)
        radius = params.get("r")
        return self.astro.schwarzschild_metrics(float(mass), float(radius) if radius is not None else None)

    def _friedmann(self, expression: str, **params: Any) -> dict[str, Any]:
        del expression
        return self.astro.friedmann_solver(**{key: params[key] for key in ("H0", "Om", "Orad", "OL", "Ok", "z") if key in params})

    def _chandrasekhar(self, expression: str, **params: Any) -> dict[str, Any]:
        del expression
        return self.astro.chandrasekhar_limit(float(params.get("mu_e", 2.0)))

    def _lane_emden(self, expression: str, base_in: int = 10, **params: Any) -> dict[str, Any]:
        n = params["n"] if "n" in params else self.parser.parse(expression, default_base=base_in)
        result = self.astro.lane_emden(float(n), params.get("xi_max"), int(params.get("points", 1000)))
        return {**result, "n": float(n)}

    def _constant(self, expression: str, base_out: int, **params: Any) -> dict[str, Any]:
        return self.constants.get(str(params.get("name", expression)), base_out)

    def _wrap_result(self, result: Any, base_out: int, params: dict[str, Any]) -> CalcResult:
        if isinstance(result, AnalysisResult):
            encoded = None
            if result.numeric is not None:
                try:
                    encoded = self.base_converter.from_decimal(result.numeric, base_out)
                except BaseConversionError:
                    encoded = None
            return CalcResult(True, value_decimal=sp.sstr(result.symbolic),
                              value_base_out=encoded,
                              latex=result.latex, data=result.data, method=result.method,
                              warnings=result.warnings)
        if isinstance(result, dict) and "base_out" in result:
            return CalcResult(True, value_decimal=result.get("decimal"), value_base_out=result["base_out"],
                              data=result, method="exact rational base conversion")
        numeric = result.get("mass_kg", result.get("luminosity_distance_Gpc")) if isinstance(result, dict) else None
        encoded = None
        if params.get("convert_numeric", False) and isinstance(numeric, (int, float, mp.mpf)):
            encoded = self.base_converter.from_decimal(numeric, base_out)
        return CalcResult(True, value_decimal=numeric, value_base_out=encoded,
                          data=result, method=result.get("method") if isinstance(result, dict) else "numerical")
