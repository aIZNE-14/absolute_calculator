import numpy as np
import pytest

from umbac.errors import PhysicsError
from umbac.quantum import QuantumEngine


def test_hermitian_eigenpairs_have_small_residuals():
    matrix = np.array([[1, 1j], [-1j, 2]], dtype=complex)
    result = QuantumEngine().matrix_eigenstates(matrix)
    assert max(result["residuals"]) < 1e-10


def test_non_hermitian_hamiltonian_is_rejected():
    with pytest.raises(PhysicsError):
        QuantumEngine().matrix_eigenstates([[1, 2], [0, 1]])


def test_harmonic_oscillator_energy_reference():
    result = QuantumEngine().schrodinger_solver("harmonic:1", -8, 8, n_points=700, n_states=3)
    assert np.allclose(result["energies"], [0.5, 1.5, 2.5], rtol=2e-4, atol=2e-4)
    assert np.allclose(np.trapezoid(result["probability_density"][0], result["x"]), 1.0)
    assert max(result["residuals"]) < 1e-8


def test_bell_state_partial_trace_entropy_and_entanglement():
    engine = QuantumEngine()
    state = engine.bell_state("phi_plus")
    reduced = engine.partial_trace(state["density_matrix"], (2, 2), (0,))
    assert np.allclose(reduced, np.eye(2) / 2)
    assert engine.quantum_entropy(reduced) == pytest.approx(1.0)
    result = engine.peres_horodecki(state["density_matrix"])
    assert result["minimum_partial_transpose_eigenvalue"] == pytest.approx(-0.5)
    assert result["negativity"] == pytest.approx(0.5)
    assert result["concurrence"] == pytest.approx(1.0)


def test_density_matrix_invariants_are_enforced():
    with pytest.raises(PhysicsError):
        QuantumEngine().quantum_entropy([[1, 2], [0, 0]])
