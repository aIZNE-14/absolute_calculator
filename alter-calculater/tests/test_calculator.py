import pytest

from calculator import convert_base, evaluate


@pytest.mark.parametrize(
    ("value", "source", "target", "expected"),
    [
        ("101101", 2, 10, "45"),
        ("-FF", 16, 10, "-255"),
        ("101.101", 2, 10, "5.625"),
        ("Z", 60, 10, "35"),
        ("a", 60, 10, "36"),
        ("x", 60, 10, "59"),
        ("59", 10, 60, "x"),
    ],
)
def test_base_conversion(value, source, target, expected):
    assert convert_base(value, source, target)["value"] == expected


def test_repeating_fraction_is_marked_as_truncated():
    result = convert_base("0.1", 10, 2)
    assert result["value"].startswith("0.000110011")
    assert result["value"].endswith("…")
    assert result["exact"] is False
    assert result["fraction"] == "1/10"


@pytest.mark.parametrize("value", ["2", "1G", "1..0", ""])
def test_invalid_digits_are_rejected(value):
    with pytest.raises(ValueError):
        convert_base(value, 2, 10)


def test_symbolic_integral_and_numeric_approximation():
    result = evaluate("integrate(sin(x)**2, x)")
    assert result["exact"] == "x/2 - sin(x)*cos(x)/2"
    assert result["approximation"] is None


def test_exact_root_and_approximation():
    result = evaluate("sqrt(2)")
    assert result["exact"] == "sqrt(2)"
    assert result["approximation"].startswith("1.41421356237")


def test_physical_constant_can_be_used():
    result = evaluate("c / 2")
    assert result["exact"] == "149896229"


def test_matrix_determinant():
    assert evaluate("det(Matrix([[1, 2], [3, 4]]))")["exact"] == "-2"


@pytest.mark.parametrize("expression", ["__import__('os')", "x.__class__", "open('file')"])
def test_unsafe_expression_is_rejected(expression):
    with pytest.raises(ValueError):
        evaluate(expression)
