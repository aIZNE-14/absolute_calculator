from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CalcConfig:
    """Calculation settings. Precision is applied locally, without changing global mpmath state."""

    dps: int = 50
    frac_digits_out: int = 40
    angle_unit: str = "rad"
    timeout_s: float = 15.0
    complex_enabled: bool = True

    def __post_init__(self) -> None:
        if self.dps < 15:
            raise ValueError("dps must be at least 15")
        if self.frac_digits_out < 0:
            raise ValueError("frac_digits_out cannot be negative")
        if self.angle_unit not in {"rad", "deg"}:
            raise ValueError("angle_unit must be 'rad' or 'deg'")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
