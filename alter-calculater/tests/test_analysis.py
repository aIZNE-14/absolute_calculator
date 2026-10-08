import sympy as sp

from umbac.analysis import MathAnalysisEngine
from umbac.errors import DivergenceError


def test_derivative_and_critical_limit():
    engine = MathAnalysisEngine()
    assert sp.simplify(engine.derive("x*sin(x)").symbolic - (sp.sin(sp.Symbol("x")) + sp.Symbol("x") * sp.cos(sp.Symbol("x")))) == 0
    assert engine.find_limit("sin(x)/x", point=0).symbolic == 1


def test_gaussian_improper_integral_reference():
    result = MathAnalysisEngine().integrate("exp(-x**2)", limits=(0, sp.oo))
    assert sp.simplify(result.symbolic - sp.sqrt(sp.pi) / 2) == 0


def test_logarithmic_improper_integral_is_reported_divergent():
    import pytest

    with pytest.raises(DivergenceError):
        MathAnalysisEngine().integrate("1/x", limits=(1, sp.oo))


def test_two_sided_limit_reports_unequal_sides():
    result = MathAnalysisEngine().find_limit("1/x", point=0)
    assert result.data["two_sided_exists"] is False
    assert len(result.warnings) == 1


def test_taylor_series_has_polynomial_and_remainder():
    result = MathAnalysisEngine().taylor("exp(x)", order=5)
    assert result.symbolic == 1 + sp.Symbol("x") + sp.Symbol("x")**2/2 + sp.Symbol("x")**3/6 + sp.Symbol("x")**4/24 + sp.Symbol("x")**5/120
    assert result.data["remainder"]


def test_fourier_series_contains_expected_sine_term():
    result = MathAnalysisEngine().fourier_series("x", period=2 * sp.pi, n_terms=4)
    assert result.symbolic.has(sp.sin(sp.Symbol("x")))


def test_harmonic_oscillator_ode_has_general_solution():
    result = MathAnalysisEngine().solve_ode("Derivative(y(x), x, 2) + y(x)")
    text = sp.sstr(result.symbolic)
    assert "sin" in text and "cos" in text
