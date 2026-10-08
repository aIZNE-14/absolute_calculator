from __future__ import annotations

from typing import Any

import numpy as np
from scipy import constants, integrate

from umbac.constants import CONSTANTS, Quantity
from umbac.errors import PhysicsError


class AstrophysicsEngine:
    """Selected, unit-explicit calculations from relativistic and stellar astrophysics."""

    def schwarzschild_metrics(self, mass: float | Quantity, r: float | Quantity | None = None) -> dict[str, Any]:
        """Return Schwarzschild radius and exterior static-observer metrics in SI units."""
        mass_kg = mass.to_si() if isinstance(mass, Quantity) else float(mass)
        if mass_kg <= 0:
            raise PhysicsError("Масса должна быть положительной.")
        gravitational_constant = constants.G
        speed_of_light = constants.c
        radius = 2 * gravitational_constant * mass_kg / speed_of_light ** 2
        result: dict[str, Any] = {
            "schwarzschild_radius_m": radius,
            "photon_sphere_radius_m": 1.5 * radius,
            "isco_radius_m": 3 * radius,
            "hawking_temperature_K": constants.hbar * speed_of_light ** 3 /
                                      (8 * np.pi * gravitational_constant * mass_kg * constants.k),
        }
        if r is None:
            return result
        radius_at_r = r.to_si() if isinstance(r, Quantity) else float(r)
        if radius_at_r <= radius:
            raise PhysicsError("r <= r_s: метрика статического наблюдателя вне области определения.")
        lapse = np.sqrt(1 - radius / radius_at_r)
        result.update({"r_m": radius_at_r, "gravitational_redshift": 1 / lapse - 1,
                       "proper_time_per_coordinate_time": lapse})
        if radius_at_r > 1.5 * radius:
            result["circular_orbit_local_speed_m_s"] = speed_of_light * np.sqrt(
                radius / (2 * (radius_at_r - radius))
            )
        else:
            result["circular_orbit_local_speed_m_s"] = None
            result["circular_orbit_warning"] = "Круговая массивная орбита невозможна внутри фотонной сферы r=1.5 r_s."
        return result

    def friedmann_solver(self, H0: float = 70.0, Om: float = 0.3, Orad: float = 0.0,
                         OL: float = 0.7, Ok: float | None = None, z: float = 1.0) -> dict[str, Any]:
        """Compute expansion rate, age, and radial/transverse luminosity distances in a FLRW model."""
        if H0 <= 0 or z < 0 or min(Om, Orad, OL) < 0:
            raise PhysicsError("H0 должен быть положительным, z неотрицательным, плотности неотрицательными.")
        curvature = 1 - Om - Orad - OL if Ok is None else float(Ok)
        h0_si = H0 * 1000 / (constants.parsec * 1e6)
        redshift = float(z)

        def expansion(a: float) -> float:
            squared = Orad / a ** 4 + Om / a ** 3 + curvature / a ** 2 + OL
            if squared <= 0:
                raise PhysicsError("H(a)^2 <= 0: параметры не дают расширяющуюся модель на этом интервале.")
            return np.sqrt(squared)

        def inverse_e(z_value: float) -> float:
            return 1 / expansion(1 / (1 + z_value))

        integral_distance, distance_error = integrate.quad(inverse_e, 0, redshift, epsabs=1e-10, epsrel=1e-10)
        if curvature > 1e-10:
            transverse_integral = np.sinh(np.sqrt(curvature) * integral_distance) / np.sqrt(curvature)
        elif curvature < -1e-10:
            transverse_integral = np.sin(np.sqrt(-curvature) * integral_distance) / np.sqrt(-curvature)
        else:
            transverse_integral = integral_distance
        hubble_z = h0_si * expansion(1 / (1 + redshift))
        comoving = constants.c / h0_si * transverse_integral
        luminosity = (1 + redshift) * comoving
        angular = comoving / (1 + redshift)
        age_integral, _ = integrate.quad(lambda a: 1 / (a * expansion(a)), 0, 1 / (1 + redshift),
                                          epsabs=1e-9, epsrel=1e-9, limit=300)
        hubble_distance_mpc = constants.c / h0_si / (constants.parsec * 1e6)
        return {"H_z_km_s_Mpc": hubble_z * constants.parsec * 1e6 / 1000,
                "scale_factor": 1 / (1 + redshift), "age_Gyr": age_integral / h0_si / (1e9 * 365.25 * 86400),
                "comoving_distance_Gpc": comoving / (constants.parsec * 1e9),
                "transverse_comoving_distance_Gpc": comoving / (constants.parsec * 1e9),
                "angular_diameter_distance_Gpc": angular / (constants.parsec * 1e9),
                "luminosity_distance_Gpc": luminosity / (constants.parsec * 1e9),
                "critical_density_kg_m3": 3 * h0_si ** 2 / (8 * np.pi * constants.G),
                "curvature_Omega_k": curvature, "quadrature_error_dimensionless": distance_error,
                "Hubble_distance_Mpc": hubble_distance_mpc}

    def lane_emden(self, n: float, xi_max: float | None = None, points: int = 2000) -> dict[str, Any]:
        """Integrate the Lane–Emden equation and stop at its first positive zero when present."""
        if n < 0 or points < 100:
            raise PhysicsError("Индекс политропы n должен быть >= 0, points >= 100.")
        if n == 0:
            xi1 = np.sqrt(6.0)
            xi = np.linspace(0, xi1, points)
            theta = 1 - xi ** 2 / 6
            derivative = -xi / 3
            return {"xi": xi, "theta": theta, "derivative": derivative,
                    "xi1": xi1, "omega_n": 2 * xi1}
        if n == 1:
            xi1 = np.pi
            xi = np.linspace(1e-8, xi1, points)
            theta = np.sin(xi) / xi
            derivative = (xi * np.cos(xi) - np.sin(xi)) / xi ** 2
            theta[0], derivative[0] = 1.0, 0.0
            return {"xi": xi, "theta": theta, "derivative": derivative,
                    "xi1": xi1, "omega_n": xi1}
        if n == 5:
            maximum = float(xi_max or 30.0)
            xi = np.linspace(0, maximum, points)
            theta = (1 + xi ** 2 / 3) ** -0.5
            derivative = -xi / 3 * (1 + xi ** 2 / 3) ** -1.5
            return {"xi": xi, "theta": theta, "derivative": derivative,
                    "xi1": float("inf"), "omega_n": np.sqrt(3.0)}
        epsilon = 1e-6
        maximum = float(xi_max or 50.0)
        initial = (1 - epsilon ** 2 / 6, -epsilon / 3 + n * epsilon ** 3 / 30)

        def equation(xi_value: float, state: np.ndarray) -> tuple[float, float]:
            theta, derivative = state
            return derivative, -2 * derivative / xi_value - max(theta, 0.0) ** n

        def first_zero(_xi: float, state: np.ndarray) -> float:
            del _xi
            return state[0]

        first_zero.terminal = True
        first_zero.direction = -1
        solution = integrate.solve_ivp(equation, (epsilon, maximum), initial, method="DOP853",
                                       dense_output=True, events=first_zero, rtol=1e-10, atol=1e-12,
                                       max_step=0.05)
        endpoint = float(solution.t_events[0][0]) if len(solution.t_events[0]) else maximum
        xi = np.linspace(epsilon, endpoint, points)
        theta, derivative = solution.sol(xi)
        xi1 = endpoint if len(solution.t_events[0]) else None
        omega_n = -endpoint ** 2 * float(solution.sol(endpoint)[1]) if xi1 is not None else None
        return {"xi": xi, "theta": theta, "derivative": derivative,
                "xi1": xi1, "omega_n": omega_n, "success": solution.success}

    def chandrasekhar_limit(self, mu_e: float = 2.0) -> dict[str, Any]:
        """Derive the n=3 polytropic mass from the relativistic electron-degeneracy equation of state."""
        if mu_e <= 0:
            raise PhysicsError("mu_e должен быть положительным.")
        polytrope = self.lane_emden(3.0, xi_max=10.0, points=1000)
        polytropic_constant = (1 / 8) * (3 / np.pi) ** (1 / 3) * constants.h * constants.c / (
            mu_e * constants.m_u
        ) ** (4 / 3)
        omega3 = float(polytrope["omega_n"])
        mass_kg = 4 * np.pi * omega3 * (polytropic_constant / (np.pi * constants.G)) ** 1.5
        mass_ratio = mass_kg / CONSTANTS["M_sun"].value
        approximate_ratio = 5.83 / mu_e ** 2
        return {"mass_solar": mass_ratio, "mass_kg": mass_kg, "mu_e": mu_e,
                "polytrope_n": 3, "xi1": polytrope["xi1"], "omega3": omega3,
                "polytropic_K_SI": polytropic_constant,
                "approximate_mass_solar": approximate_ratio,
                "method": "M=4π ω₃ (K/(πG))^(3/2), K=(1/8)(3/π)^(1/3)hc/(μₑmᵤ)^(4/3)",
                "note": "Идеальный холодный ультрарелятивистский электронный газ; оболочка, вращение, температура и Coulomb corrections не включены."}

    def hydrostatic_equilibrium(self, profile: dict[str, Any]) -> dict[str, Any]:
        """Compare supplied pressure gradients with dP/dr=-Gm(r)rho(r)/r^2."""
        radius = np.asarray(profile.get("r"), dtype=float)
        pressure = np.asarray(profile.get("P"), dtype=float)
        mass = np.asarray(profile.get("m"), dtype=float)
        density = np.asarray(profile.get("rho"), dtype=float)
        if radius.ndim != 1 or len(radius) < 3 or any(values.shape != radius.shape for values in (pressure, mass, density)):
            raise PhysicsError("Профиль должен содержать массивы r, P, m, rho одинаковой длины (>=3).")
        if np.any(radius <= 0) or np.any(np.diff(radius) <= 0) or np.any(mass < 0) or np.any(density < 0):
            raise PhysicsError("Радиусы должны возрастать; масса и плотность неотрицательны.")
        observed = np.gradient(pressure, radius, edge_order=2)
        expected = -constants.G * mass * density / radius ** 2
        scale = np.maximum(np.abs(expected), np.max(np.abs(expected)) * 1e-14 + 1e-300)
        relative = np.abs(observed - expected) / scale
        return {"r": radius, "dP_dr_observed": observed, "dP_dr_expected": expected,
                "relative_error": relative, "max_relative_error": float(np.max(relative))}
