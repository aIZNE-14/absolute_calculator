from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp
from scipy import constants as scipy_constants

from umbac.base_converter import BaseConverter


@dataclass(frozen=True, slots=True)
class Constant:
    """A physical constant with SI unit, standard uncertainty, and source."""

    value: float
    unit: str
    uncertainty: float | None
    source: str


CONSTANTS = {
    "c": Constant(scipy_constants.c, "m s^-1", 0.0, "SI exact definition"),
    "G": Constant(scipy_constants.G, "m^3 kg^-1 s^-2", scipy_constants.physical_constants["Newtonian constant of gravitation"][2], "CODATA 2022"),
    "h": Constant(scipy_constants.h, "J s", 0.0, "SI exact definition"),
    "hbar": Constant(scipy_constants.hbar, "J s", 0.0, "SI exact definition"),
    "k_B": Constant(scipy_constants.k, "J K^-1", 0.0, "SI exact definition"),
    "sigma_SB": Constant(scipy_constants.Stefan_Boltzmann, "W m^-2 K^-4", 0.0, "CODATA"),
    "e_charge": Constant(scipy_constants.e, "C", 0.0, "SI exact definition"),
    "m_e": Constant(scipy_constants.m_e, "kg", scipy_constants.physical_constants["electron mass"][2], "CODATA 2022"),
    "m_p": Constant(scipy_constants.m_p, "kg", scipy_constants.physical_constants["proton mass"][2], "CODATA 2022"),
    "AU": Constant(149597870700.0, "m", 0.0, "IAU 2012"),
    "M_sun": Constant(1.98847e30, "kg", 0.0, "IAU 2015 nominal"),
    "R_sun": Constant(6.957e8, "m", 0.0, "IAU 2015 nominal"),
    "L_sun": Constant(3.828e26, "W", 0.0, "IAU 2015 nominal"),
    "pc": Constant(scipy_constants.parsec, "m", 0.0, "IAU"),
    "ly": Constant(scipy_constants.light_year, "m", 0.0, "IAU"),
}


class Constants:
    """Look up SI constants and optionally encode their values in another base."""

    def get(self, name: str, base_out: int | None = None) -> dict[str, object]:
        if name not in CONSTANTS:
            raise KeyError(f"Неизвестная физическая константа: {name}")
        constant = CONSTANTS[name]
        result: dict[str, object] = {
            "name": name, "value": constant.value, "unit": constant.unit,
            "uncertainty": constant.uncertainty, "source": constant.source,
        }
        if base_out is not None:
            result["value_base"] = BaseConverter().from_decimal(mp.mpf(str(constant.value)), base_out)
        return result


@dataclass(frozen=True, slots=True)
class Quantity:
    """A scalar physical quantity normalized to SI by an explicit unit factor."""

    value: float
    unit: str

    def to_si(self) -> float:
        factors = {"m": 1.0, "km": 1e3, "cm": 1e-2, "kg": 1.0, "g": 1e-3,
                   "s": 1.0, "day": 86400.0, "yr": 31557600.0, "K": 1.0,
                   "M_sun": CONSTANTS["M_sun"].value, "R_sun": CONSTANTS["R_sun"].value,
                   "AU": CONSTANTS["AU"].value, "pc": scipy_constants.parsec,
                   "Mpc": scipy_constants.parsec * 1e6}
        if self.unit not in factors:
            raise ValueError(f"Единица {self.unit!r} не поддерживается.")
        return self.value * factors[self.unit]
