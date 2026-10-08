from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

import mpmath as mp
import sympy as sp
from umbac.config import CalcConfig
from umbac.errors import DivergenceError, SolverError
from umbac.expr_parser import DEFAULT_FUNCTIONS, SafeExpressionParser


@dataclass(slots=True)
class AnalysisResult:
    """A symbolic or numerical result with method and caveats."""

    symbolic: Any
    latex: str
    numeric: Any = None
    method: str = "symbolic"
    warnings: list[str] = field(default_factory=list)
    data: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"symbolic": sp.sstr(self.symbolic), "latex": self.latex,
                "numeric": str(self.numeric) if self.numeric is not None else None,
                "method": self.method, "warnings": self.warnings, "data": self.data}


class MathAnalysisEngine:
    """Symbolic and numerical tools for single-variable and selected ODE problems."""

    def __init__(self, config: CalcConfig | None = None, parser: SafeExpressionParser | None = None) -> None:
        self.config = config or CalcConfig()
        if parser is None:
            functions = dict(DEFAULT_FUNCTIONS)
            if self.config.angle_unit == "deg":
                functions.update({
                    "sin": lambda value: sp.sin(value * sp.pi / 180),
                    "cos": lambda value: sp.cos(value * sp.pi / 180),
                    "tan": lambda value: sp.tan(value * sp.pi / 180),
                    "asin": lambda value: sp.asin(value) * 180 / sp.pi,
                    "acos": lambda value: sp.acos(value) * 180 / sp.pi,
                    "atan": lambda value: sp.atan(value) * 180 / sp.pi,
                })
            self.parser = SafeExpressionParser(
                {"I": sp.I} if self.config.complex_enabled else {}, functions,
                blocked_names=set() if self.config.complex_enabled else {"I"},
            )
        else:
            self.parser = parser

    @staticmethod
    def _variable(variable: str | sp.Symbol) -> sp.Symbol:
        return variable if isinstance(variable, sp.Symbol) else sp.Symbol(variable)

    def _result(self, value: Any, method: str = "symbolic", warnings_: list[str] | None = None, data: Any = None) -> AnalysisResult:
        numeric = None
        if isinstance(value, sp.Basic) and not value.free_symbols and value.is_finite is not False:
            try:
                with mp.workdps(self.config.dps):
                    numeric = mp.mpf(str(value.evalf(self.config.dps)))
            except (ValueError, TypeError):
                numeric = None
        return AnalysisResult(value, sp.latex(value), numeric, method, warnings_ or [], data)

    def derive(self, expr: str | sp.Expr, var: str | sp.Symbol = "x", order: int = 1,
               point: Any = None) -> AnalysisResult:
        """Differentiate symbolically; optional point substitutes after differentiation."""
        if order < 0:
            raise ValueError("Порядок производной не может быть отрицательным.")
        variable = self._variable(var)
        expression = self.parser.parse(expr) if isinstance(expr, str) else expr
        value = sp.diff(expression, variable, order)
        warnings_ = []
        if expression.has(sp.Abs, sp.sign):
            warnings_.append("Производная может быть негладкой в точках разрыва или излома.")
        if point is not None:
            value = value.subs(variable, self.parser.parse(str(point)))
        return self._result(value, warnings_=warnings_)

    def integrate(self, expr: str | sp.Expr, var: str | sp.Symbol = "x",
                   limits: Sequence[Any] | None = None) -> AnalysisResult:
        """Compute an indefinite/definite integral, with numeric fallback for finite bounds."""
        variable = self._variable(var)
        expression = self.parser.parse(expr) if isinstance(expr, str) else expr
        if limits is None:
            return self._result(sp.integrate(expression, variable))
        if len(limits) != 2:
            raise ValueError("Для определённого интеграла задайте пару (нижняя, верхняя).")
        lower, upper = [self.parser.parse(str(bound)) if isinstance(bound, str) else bound for bound in limits]
        symbolic = sp.integrate(expression, (variable, lower, upper))
        if symbolic in (sp.oo, -sp.oo, sp.zoo) or symbolic.is_finite is False:
            raise DivergenceError("Несобственный интеграл расходится.", "Проверьте поведение подынтегральной функции на бесконечности и в особенностях.")
        if not symbolic.has(sp.Integral):
            return self._result(symbolic)
        try:
            function = sp.lambdify(variable, expression, modules="mpmath")
            with mp.workdps(self.config.dps):
                low = self._mp_bound(lower)
                high = self._mp_bound(upper)
                result = mp.quad(function, [low, high])
                if not mp.isfinite(result):
                    raise DivergenceError("Интеграл расходится.")
                if lower not in (sp.oo, -sp.oo) and upper not in (sp.oo, -sp.oo):
                    return self._result(sp.Float(str(result), self.config.dps), "numeric")
                return self._result(sp.Float(str(result), self.config.dps), "numeric",
                                    ["Несобственный интеграл оценён численно; проверьте сходимость отдельно."])
        except DivergenceError:
            raise
        except Exception as error:
            raise DivergenceError(f"Интеграл не удалось вычислить: {error}", "Разбейте область в точках особенностей и проверьте сходимость.") from error

    @staticmethod
    def _mp_bound(bound: Any) -> Any:
        if bound == sp.oo:
            return mp.inf
        if bound == -sp.oo:
            return -mp.inf
        return mp.mpf(str(bound))

    def find_limit(self, expr: str | sp.Expr, var: str | sp.Symbol = "x", point: Any = 0,
                   direction: str = "+-") -> AnalysisResult:
        """Compute one-sided or two-sided limits and report unequal one-sided limits."""
        if direction not in {"+", "-", "+-"}:
            raise ValueError("direction должен быть '+', '-' или '+-'.")
        variable = self._variable(var)
        expression = self.parser.parse(expr) if isinstance(expr, str) else expr
        target = self.parser.parse(str(point)) if isinstance(point, str) else point
        if direction != "+-":
            value = sp.limit(expression, variable, target, dir=direction)
            return self._result(value)
        left = sp.limit(expression, variable, target, dir="-")
        right = sp.limit(expression, variable, target, dir="+")
        if sp.simplify(left - right) == 0:
            return self._result(left)
        return self._result({"left": left, "right": right}, warnings_=["Двусторонний предел не существует: односторонние пределы различаются."], data={"two_sided_exists": False})

    def taylor(self, expr: str | sp.Expr, var: str | sp.Symbol = "x", point: Any = 0,
               order: int = 5) -> AnalysisResult:
        """Return a Taylor polynomial and a numerical Lagrange-remainder estimate when possible."""
        if order < 0:
            raise ValueError("Порядок ряда не может быть отрицательным.")
        variable = self._variable(var)
        expression = self.parser.parse(expr) if isinstance(expr, str) else expr
        center = self.parser.parse(str(point)) if isinstance(point, str) else point
        series = sp.series(expression, variable, center, order + 1)
        polynomial = sp.expand(series.removeO())
        remainder = series.getO()
        warning_list = []
        remainder_bound = None
        try:
            radius = mp.mpf("0.1")
            derivative = sp.diff(expression, variable, order + 1)
            derivative_fn = sp.lambdify(variable, derivative, modules="mpmath")
            center_mp = mp.mpf(str(center))
            maximum = max(abs(derivative_fn(center_mp - radius)), abs(derivative_fn(center_mp + radius)))
            remainder_bound = str(maximum * radius ** (order + 1) / mp.factorial(order + 1))
        except Exception:
            warning_list.append("Численная оценка остатка Лагранжа не получена.")
        return self._result(polynomial, warnings_=warning_list,
                            data={"remainder": sp.sstr(remainder), "lagrange_bound_radius_0_1": remainder_bound})

    def fourier_series(self, expr: str | sp.Expr, var: str | sp.Symbol = "x",
                       period: Any = 2 * sp.pi, n_terms: int = 10) -> AnalysisResult:
        """Compute a truncated Fourier series on the symmetric interval of the given period."""
        if n_terms < 1:
            raise ValueError("n_terms должен быть положительным.")
        variable = self._variable(var)
        expression = self.parser.parse(expr) if isinstance(expr, str) else expr
        length = sp.sympify(period) / 2
        series = sp.fourier_series(expression, (variable, -length, length))
        polynomial = sp.expand(series.truncate(n_terms + 1))
        error = None
        warnings_ = []
        try:
            error = sp.sqrt(sp.integrate((expression - polynomial) ** 2, (variable, -length, length)) / period)
        except Exception:
            warnings_.append("Среднеквадратичная ошибка не найдена в замкнутой форме.")
        if expression.has(sp.Piecewise, sp.sign, sp.Heaviside):
            warnings_.append("В окрестности скачков возможен эффект Гиббса.")
        return self._result(polynomial, warnings_=warnings_, data={"rms_error": sp.sstr(error) if error is not None else None})

    def solve_ode(self, eq: str | sp.Expr, func: Any = "y", ics: dict[Any, Any] | None = None,
                  order: int | None = None) -> AnalysisResult:
        """Solve an ODE represented as a SymPy equation/expression with supplied initial conditions."""
        equation = self.parser.parse(eq) if isinstance(eq, str) else eq
        if not isinstance(equation, sp.Equality):
            equation = sp.Eq(equation, 0)
        if not equation.has(sp.Derivative):
            raise SolverError("В выражении не найдены производные неизвестной функции.", "Задайте уравнение с sympy.Function и Derivative.")
        derivative_orders = [derivative.derivative_count for derivative in equation.atoms(sp.Derivative)]
        actual_order = max(derivative_orders, default=0)
        if order is not None and order != actual_order:
            raise SolverError(f"Уравнение имеет порядок {actual_order}, а запрошен порядок {order}.")
        target = None
        applied_functions = equation.atoms(sp.Function)
        if isinstance(func, str):
            target = next((applied for applied in applied_functions
                           if applied.func.__name__ == func), None)
        else:
            target = next((applied for applied in applied_functions if applied.func == func), func)
        solution = sp.dsolve(equation, func=target, ics=ics) if target is not None else sp.dsolve(equation, ics=ics)
        return self._result(solution)

    def solve_pde(self, eq: Any, func: Any = None, **kwargs: Any) -> AnalysisResult:
        """Try SymPy separation/first-order PDE solvers; unsupported PDEs fail explicitly."""
        try:
            solution = sp.pdsolve(eq, func=func, **kwargs)
        except (NotImplementedError, ValueError) as error:
            raise SolverError("PDE общего вида не поддерживаются.", "Доступны только уравнения, решаемые SymPy pdsolve.") from error
        return self._result(solution)

    def simplify_check(self, first: str | sp.Expr, second: str | sp.Expr) -> AnalysisResult:
        """Check symbolic identity, then compare deterministic random numeric samples."""
        left = self.parser.parse(first) if isinstance(first, str) else first
        right = self.parser.parse(second) if isinstance(second, str) else second
        difference = sp.simplify(left - right)
        if difference == 0:
            return self._result(sp.true, data={"symbolic_equal": True, "numeric_checks": 0})
        symbols = sorted(difference.free_symbols, key=str)
        if not symbols:
            return self._result(sp.false, data={"symbolic_equal": False, "numeric_checks": 0})
        generator = random.Random(1729)
        checked = 0
        for _ in range(20):
            substitutions = {symbol: sp.Rational(generator.randint(1, 17), generator.randint(1, 17)) for symbol in symbols}
            try:
                value = complex(difference.subs(substitutions).evalf())
            except (ValueError, TypeError, ZeroDivisionError):
                continue
            checked += 1
            if abs(value) > 1e-10:
                return self._result(sp.false, data={"symbolic_equal": False, "numeric_checks": checked})
        return self._result(sp.true, warnings_=["Численная выборка не доказывает тождество."],
                            data={"symbolic_equal": None, "numeric_checks": checked})
