from umbac import CalcConfig, UniversalAdvancedCalculator


def test_facade_formats_and_dispatches_base_operations():
    calculator = UniversalAdvancedCalculator()
    result = calculator.compute("255", "base", 10, 16)
    assert result.ok
    assert result.value_base_out == "FF"
    assert calculator.compute("101 + 1", "analysis.evaluate", base_in=2).value_decimal == "6"
    assert calculator.compute("1F", "analysis.evaluate", base_in=16).value_decimal == "31"
    assert calculator.compute("ff", "analysis.evaluate", base_in=16).value_decimal == "255"
    assert calculator.compute("1.8", "analysis.evaluate", base_in=16).value_decimal == "3/2"
    assert calculator.compute("3.14*x", "analysis.evaluate").value_decimal == "157*x/50"


def test_facade_returns_structured_errors_not_fake_answers():
    result = UniversalAdvancedCalculator().compute("", "quantum.schrodinger", n_points=4)
    assert not result.ok
    assert result.error
    assert "error" in result.to_json()
    assert not UniversalAdvancedCalculator().compute("1/0", "analysis.evaluate").ok


def test_calc_config_validation_and_json_serialization():
    config = CalcConfig(dps=60, angle_unit="deg")
    calculator = UniversalAdvancedCalculator(config)
    result = calculator.compute("sin(30)", "analysis.evaluate", base_in=10)
    assert result.ok
    assert CalcConfig().dps == 50
    assert '"ok": true' in result.to_json()


def test_angle_unit_and_complex_mode_are_enforced():
    degrees = UniversalAdvancedCalculator(CalcConfig(angle_unit="deg"))
    assert degrees.compute("sin(30)", "analysis.evaluate").value_decimal == "1/2"
    real_only = UniversalAdvancedCalculator(CalcConfig(complex_enabled=False))
    assert not real_only.compute("sqrt(-1)", "analysis.evaluate").ok