import numpy as np
import pytest

from umbac.astro import AstrophysicsEngine
from umbac.errors import PhysicsError
from umbac.quantum import QuantumEngine


def test_solar_schwarzschild_radius_and_horizon_rejection():
    engine = AstrophysicsEngine()
    radius = engine.schwarzschild_metrics(1.98847e30)["schwarzschild_radius_m"]
    assert radius == pytest.approx(2953, rel=0.01)
    with pytest.raises(PhysicsError):
        engine.schwarzschild_metrics(1e30, r=1.0)


def test_friedmann_luminosity_distance_reference():
    result = AstrophysicsEngine().friedmann_solver(H0=70, Om=0.3, OL=0.7, z=1)
    assert result["luminosity_distance_Gpc"] == pytest.approx(6.6, abs=0.15)


def test_lane_emden_analytic_indices():
    engine = AstrophysicsEngine()
    assert engine.lane_emden(0)["xi1"] == pytest.approx(6 ** 0.5)
    assert engine.lane_emden(1)["xi1"] == pytest.approx(3.141592653589793)
    assert engine.lane_emden(5)["xi1"] == float("inf")


def test_chandrasekhar_mass_reference():
    result = AstrophysicsEngine().chandrasekhar_limit(2)
    assert result["mass_solar"] == pytest.approx(1.4563, abs=0.002)
    assert result["xi1"] == pytest.approx(6.89685, abs=1e-4)
    assert result["omega3"] == pytest.approx(2.01824, abs=1e-4)
    assert result["approximate_mass_solar"] == pytest.approx(1.4575)


def test_ppt_sufficiency_is_explicit_for_higher_dimensions():
    result = QuantumEngine().peres_horodecki(np.eye(8) / 8, (2, 4))
    assert result["persistence_of_entanglement_criterion_sufficient"] is False
