from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from scipy import constants
from scipy.linalg import eigh, eigh_tridiagonal

from umbac.errors import PhysicsError


class QuantumEngine:
    """Numerical finite-dimensional and one-dimensional stationary quantum tools."""

    def __init__(self, units: str = "atomic") -> None:
        if units not in {"atomic", "SI"}:
            raise PhysicsError("Единицы должны быть atomic или SI.")
        self.units = units
        self.hbar = 1.0 if units == "atomic" else constants.hbar
        self.default_mass = 1.0 if units == "atomic" else constants.m_e

    def matrix_eigenstates(self, matrix: Any, hermitian_check: bool = True) -> dict[str, Any]:
        """Diagonalize a Hamiltonian and verify eigenpair residuals and orthonormality."""
        hamiltonian = np.asarray(matrix, dtype=np.complex128)
        if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
            raise PhysicsError("Гамильтониан должен быть квадратной матрицей.")
        if hermitian_check and not np.allclose(hamiltonian, hamiltonian.conj().T, atol=1e-10):
            raise PhysicsError("Матрица не эрмитова; eigh применять нельзя.")
        values, vectors = eigh(hamiltonian) if hermitian_check else np.linalg.eig(hamiltonian)
        order = np.argsort(values.real)
        values, vectors = values[order], vectors[:, order]
        residuals = [float(np.linalg.norm(hamiltonian @ vectors[:, i] - values[i] * vectors[:, i]))
                     for i in range(len(values))]
        return {"values": values.real if hermitian_check else values,
                "vectors": vectors, "residuals": residuals,
                "degenerate_pairs": [(i, j) for i in range(len(values)) for j in range(i + 1, len(values))
                                     if abs(values[i] - values[j]) < 1e-10]}

    def operators(self, grid: Sequence[float], mass: float | None = None) -> dict[str, Any]:
        """Build coordinate, fourth-order momentum, and finite-difference Hamiltonian operators."""
        points = np.asarray(grid, dtype=float)
        if points.ndim != 1 or len(points) < 5 or not np.all(np.diff(points) > 0):
            raise PhysicsError("Сетка должна содержать минимум пять строго возрастающих точек.")
        spacings = np.diff(points)
        if not np.allclose(spacings, spacings[0], rtol=1e-8, atol=1e-12):
            raise PhysicsError("Оператор конечных разностей требует равномерную сетку.")
        particle_mass = self.default_mass if mass is None else float(mass)
        if particle_mass <= 0:
            raise PhysicsError("Масса должна быть положительной.")
        step = float(spacings[0])
        derivative = np.gradient(np.eye(len(points)), step, axis=0, edge_order=2)
        coordinate = np.diag(points)
        momentum = -1j * self.hbar * derivative
        identity = np.eye(len(points), dtype=np.complex128)
        return {"x": coordinate, "p": momentum,
                "kinetic": -(self.hbar ** 2 / (2 * particle_mass)) * derivative @ derivative,
                "grid": points, "identity": identity}

    def schrodinger_solver(self, potential: str | Callable[[np.ndarray], Any], x_min: float, x_max: float,
                           n_points: int = 2000, n_states: int = 6, mass: float | None = None) -> dict[str, Any]:
        """Solve the 1-D time-independent equation with Dirichlet boundaries and a tridiagonal Hamiltonian."""
        if not x_min < x_max or n_points < 5 or not 1 <= n_states < n_points - 2:
            raise PhysicsError("Проверьте границы, n_points и число собственных состояний.")
        particle_mass = self.default_mass if mass is None else float(mass)
        if particle_mass <= 0:
            raise PhysicsError("Масса должна быть положительной.")
        grid = np.linspace(x_min, x_max, n_points)
        dx = grid[1] - grid[0]
        interior = grid[1:-1]
        values = self._potential(potential, interior, particle_mass)
        if values.shape != interior.shape or not np.all(np.isfinite(values)):
            raise PhysicsError("Потенциал должен вернуть конечное значение на каждой точке сетки.")
        coefficient = self.hbar ** 2 / (2 * particle_mass * dx ** 2)
        diagonal = 2 * coefficient + values
        off_diagonal = np.full(len(interior) - 1, -coefficient)
        energies, interior_states = eigh_tridiagonal(
            diagonal, off_diagonal, select="i", select_range=(0, n_states - 1),
            check_finite=True,
        )
        states = np.zeros((n_states, n_points), dtype=float)
        states[:, 1:-1] = interior_states.T
        for index in range(n_states):
            norm = np.sqrt(np.trapezoid(np.abs(states[index]) ** 2, grid))
            states[index] /= norm
            first_nonzero = np.flatnonzero(np.abs(states[index]) > 1e-14)
            if first_nonzero.size and states[index, first_nonzero[0]] < 0:
                states[index] *= -1
        residuals = []
        for index in range(n_states):
            vector = interior_states[:, index]
            left_neighbor = np.concatenate(([0.0], off_diagonal * vector[:-1]))
            right_neighbor = np.concatenate((off_diagonal * vector[1:], [0.0]))
            residual = diagonal * vector + left_neighbor + right_neighbor - energies[index] * vector
            residuals.append(float(np.linalg.norm(residual)))
        return {"energies": energies, "x": grid, "wavefunctions": states,
                "probability_density": np.abs(states) ** 2, "residuals": residuals,
                "units": self.units, "method": "finite-difference / eigh_tridiagonal"}

    def _potential(self, potential: str | Callable[[np.ndarray], Any], x: np.ndarray, mass: float) -> np.ndarray:
        if callable(potential):
            return np.asarray(potential(x), dtype=float)
        if not isinstance(potential, str):
            raise PhysicsError("Потенциал должен быть callable или именованной моделью.")
        parts = potential.split(":")
        name = parts[0]
        parameter = float(parts[1]) if len(parts) > 1 else 1.0
        if name == "infinite_well":
            result = np.zeros_like(x)
        elif name == "harmonic":
            result = 0.5 * mass * parameter ** 2 * x ** 2
        elif name == "finite_well":
            width = float(parts[2]) if len(parts) > 2 else 1.0
            result = np.where(np.abs(x) < width / 2, -parameter, 0.0)
        elif name == "step":
            result = np.where(x < 0, 0.0, parameter)
        else:
            raise PhysicsError(f"Неизвестный потенциал {name!r}.", "Доступны infinite_well, finite_well:V0:width, harmonic:omega и step:V0.")
        return np.asarray(result, dtype=float)

    @staticmethod
    def normalize(psi: Sequence[complex], x: Sequence[float]) -> np.ndarray:
        """Normalize a wavefunction so that its integrated probability is one."""
        wavefunction, grid = np.asarray(psi, dtype=np.complex128), np.asarray(x, dtype=float)
        norm = float(np.trapezoid(np.abs(wavefunction) ** 2, grid))
        if norm <= 0 or not np.isfinite(norm):
            raise PhysicsError("Волновая функция имеет нулевую или некорректную норму.")
        return wavefunction / np.sqrt(norm)

    @staticmethod
    def probability_density(psi: Sequence[complex]) -> np.ndarray:
        """Return the Born probability density |psi|^2."""
        return np.abs(np.asarray(psi, dtype=np.complex128)) ** 2

    @staticmethod
    def expectation(operator: Any, psi: Sequence[complex], x: Sequence[float] | None = None) -> complex:
        """Evaluate a normalized-state expectation value for a matrix or sampled operator."""
        state = np.asarray(psi, dtype=np.complex128)
        applied = np.asarray(operator) @ state
        integrand = np.conjugate(state) * applied
        return complex(np.trapezoid(integrand, x)) if x is not None else complex(np.vdot(state, applied))

    @staticmethod
    def uncertainty(operator: Any, psi: Sequence[complex]) -> float:
        """Compute sqrt(<A^2> - <A>^2) for a Hermitian matrix operator."""
        matrix = np.asarray(operator, dtype=np.complex128)
        state = np.asarray(psi, dtype=np.complex128)
        mean = np.vdot(state, matrix @ state)
        second_moment = np.vdot(state, matrix @ matrix @ state)
        variance = float(np.real(second_moment - mean ** 2))
        if variance < -1e-10:
            raise PhysicsError("Дисперсия отрицательна; оператор или состояние некорректны.")
        return float(np.sqrt(max(variance, 0.0)))

    def quantum_entropy(self, density_matrix: Any) -> float:
        """Compute von Neumann entropy in bits after validating a density matrix."""
        rho = self._density_matrix(density_matrix)
        eigenvalues = np.linalg.eigvalsh(rho).real
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    @staticmethod
    def qubit_state(alpha: complex, beta: complex) -> dict[str, Any]:
        """Normalize a pure qubit and return its Bloch coordinates and density matrix."""
        state = np.asarray([alpha, beta], dtype=np.complex128)
        norm = np.linalg.norm(state)
        if norm <= 0:
            raise PhysicsError("Кубит не может быть нулевым вектором.")
        state /= norm
        alpha, beta = state
        rho = np.outer(state, state.conj())
        bloch = [float(2 * np.real(alpha.conjugate() * beta)),
                 float(2 * np.imag(alpha.conjugate() * beta)),
                 float(abs(alpha) ** 2 - abs(beta) ** 2)]
        return {"state": state, "density_matrix": rho, "bloch": bloch}

    @staticmethod
    def bell_state(kind: str = "phi_plus") -> dict[str, Any]:
        """Construct one of the four Bell states and its pure-state density matrix."""
        states = {
            "phi_plus": np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
            "phi_minus": np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),
            "psi_plus": np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),
            "psi_minus": np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),
        }
        if kind not in states:
            raise PhysicsError(f"Неизвестное состояние Белла: {kind}")
        state = states[kind]
        return {"kind": kind, "state": state, "density_matrix": np.outer(state, state.conj())}

    @staticmethod
    def partial_trace(rho: Any, dims: Sequence[int], keep: Sequence[int]) -> np.ndarray:
        """Trace out all subsystems except the indexed subsystems in keep."""
        matrix = np.asarray(rho, dtype=np.complex128)
        dimensions = tuple(int(size) for size in dims)
        if np.prod(dimensions) != matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
            raise PhysicsError("dims не согласуются с размером матрицы плотности.")
        retained = tuple(sorted(set(int(index) for index in keep)))
        if not retained or any(index < 0 or index >= len(dimensions) for index in retained):
            raise PhysicsError("keep должен задавать существующие подсистемы.")
        tensor = matrix.reshape(*dimensions, *dimensions)
        current_dims = list(dimensions)
        traced = [index for index in range(len(dimensions)) if index not in retained]
        for index in reversed(traced):
            tensor = np.trace(tensor, axis1=index, axis2=index + len(current_dims))
            current_dims.pop(index)
        dimension = int(np.prod(current_dims))
        return tensor.reshape(dimension, dimension)

    @staticmethod
    def _density_matrix(density_matrix: Any) -> np.ndarray:
        rho = np.asarray(density_matrix, dtype=np.complex128)
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise PhysicsError("Матрица плотности должна быть квадратной.")
        if not np.allclose(rho, rho.conj().T, atol=1e-10):
            raise PhysicsError("Матрица плотности должна быть эрмитовой.")
        if not np.isclose(np.trace(rho), 1.0, atol=1e-10):
            raise PhysicsError("След матрицы плотности должен быть равен единице.")
        if np.min(np.linalg.eigvalsh(rho)) < -1e-10:
            raise PhysicsError("Матрица плотности должна быть положительно полуопределённой.")
        return rho

    def peres_horodecki(self, rho: Any, dims: Sequence[int] = (2, 2)) -> dict[str, Any]:
        """Apply the partial-transpose PPT test; it is sufficient only for 2x2 and 2x3."""
        matrix = self._density_matrix(rho)
        if len(dims) < 2 or np.prod(dims) != matrix.shape[0]:
            raise PhysicsError("Размеры подсистем не совпадают с матрицей.")
        tensor = matrix.reshape(*dims, *dims)
        axes = list(range(2 * len(dims)))
        axes[0], axes[len(dims)] = axes[len(dims)], axes[0]
        transposed = tensor.transpose(axes).reshape(matrix.shape)
        minimum = float(np.linalg.eigvalsh(transposed).min())
        dimensions = sorted((int(size) for size in dims))
        sufficient = len(dimensions) == 2 and dimensions[0] == 2 and dimensions[1] in (2, 3)
        negativity = float(max(0.0, -np.linalg.eigvalsh(transposed).min()))
        result: dict[str, Any] = {"minimum_partial_transpose_eigenvalue": minimum,
                                  "negativity": negativity, "ppt": minimum >= -1e-10,
                                  "persistence_of_entanglement_criterion_sufficient": sufficient,
                                  "warning": None if sufficient else "Для данной размерности PPT-критерий только необходимый, не достаточный."}
        if tuple(dims) == (2, 2):
            sigma_y = np.array([[0, -1j], [1j, 0]])
            spin_flip = np.kron(sigma_y, sigma_y) @ matrix.conj() @ np.kron(sigma_y, sigma_y)
            eigenvalues = np.sort(np.sqrt(np.maximum(np.linalg.eigvals(matrix @ spin_flip).real, 0)))[::-1]
            result["concurrence"] = float(max(0.0, eigenvalues[0] - np.sum(eigenvalues[1:])))
        return result
