from __future__ import annotations

from fractions import Fraction
from typing import Any

import mpmath as mp

from umbac.config import CalcConfig
from umbac.errors import BaseConversionError


ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+/"
MIN_BASE = 2
MAX_BASE = len(ALPHABET)


class BaseConverter:
    """Convert positional rational numbers between bases 2 and 64."""

    def __init__(self, config: CalcConfig | None = None) -> None:
        self.config = config or CalcConfig()

    @staticmethod
    def validate_base(base: int) -> None:
        if not MIN_BASE <= base <= MAX_BASE:
            raise BaseConversionError(f"Основание {base} вне диапазона 2–64.", "Выберите целое основание от 2 до 64.")

    @classmethod
    def parse_fraction(cls, text: str, base: int) -> Fraction:
        cls.validate_base(base)
        value = text.strip()
        if not value:
            raise BaseConversionError("Пустая запись числа.")
        sign = -1 if value[0] == "-" else 1
        if value.startswith("-"):
            value = value[1:]
        elif value.startswith("+") and base < 63:
            value = value[1:]
        if value.count(".") > 1 or not value or value == ".":
            raise BaseConversionError("Неверный формат позиционного числа.")
        integer, separator, fractional = value.partition(".")
        integer = integer or "0"
        total = 0
        for position, character in enumerate(integer + fractional):
            digit = cls._digit(character, base, position)
            total = total * base + digit
        denominator = base ** len(fractional) if separator else 1
        if separator and not fractional:
            denominator = 1
        return sign * Fraction(total, denominator)

    @staticmethod
    def _digit(character: str, base: int, position: int) -> int:
        if base <= 36 and character.isascii() and character.isalpha():
            character = character.upper()
        digit = ALPHABET.find(character)
        if digit < 0 or digit >= base:
            raise BaseConversionError(
                f"Недопустимый знак {character!r} в позиции {position + 1} для основания {base}.",
                "Проверьте алфавит и основание исходного числа.",
            )
        return digit

    @staticmethod
    def _integer_text(value: int, base: int) -> str:
        if value == 0:
            return "0"
        digits = []
        while value:
            value, remainder = divmod(value, base)
            digits.append(ALPHABET[remainder])
        return "".join(reversed(digits))

    def to_decimal(self, text: str, base: int) -> Any:
        """Return the exact positional value rounded to configured decimal precision."""
        value = self.parse_fraction(text, base)
        with mp.workdps(self.config.dps):
            return mp.mpf(value.numerator) / value.denominator

    def from_decimal(self, value: Any, base: int, frac_digits: int | None = None) -> str:
        """Format a decimal value with round-half-up precision in the target base."""
        self.validate_base(base)
        places = self.config.frac_digits_out if frac_digits is None else frac_digits
        if places < 0:
            raise BaseConversionError("Количество дробных знаков не может быть отрицательным.")
        rational = self._as_fraction(value)
        sign = "-" if rational < 0 else ""
        rational = abs(rational)
        integer, remainder = divmod(rational.numerator, rational.denominator)
        digits = []
        for _ in range(places + 1):
            if not remainder:
                break
            remainder *= base
            digit, remainder = divmod(remainder, rational.denominator)
            digits.append(digit)
        if len(digits) > places:
            round_digit = digits.pop()
            if round_digit * 2 >= base:
                carry = 1
                for index in range(len(digits) - 1, -1, -1):
                    digits[index] += carry
                    carry, digits[index] = divmod(digits[index], base)
                integer += carry
        fractional = "".join(ALPHABET[digit] for digit in digits).rstrip("0")
        result = sign + self._integer_text(integer, base)
        return result + ("." + fractional if fractional else "")

    def from_decimal_periodic(self, value: Fraction, base: int, max_period: int = 1000) -> str:
        """Represent a rational number with a parenthesized repeating digit cycle."""
        self.validate_base(base)
        if max_period <= 0:
            raise BaseConversionError("max_period must be positive")
        sign = "-" if value < 0 else ""
        value = abs(value)
        integer, remainder = divmod(value.numerator, value.denominator)
        whole = self._integer_text(integer, base)
        if not remainder:
            return sign + whole
        positions: dict[int, int] = {}
        digits: list[int] = []
        while remainder and remainder not in positions and len(digits) < max_period:
            positions[remainder] = len(digits)
            remainder *= base
            digit, remainder = divmod(remainder, value.denominator)
            digits.append(digit)
        prefix = "".join(ALPHABET[digit] for digit in digits)
        if remainder == 0:
            return sign + whole + ("." + prefix if prefix else "")
        if remainder in positions:
            start = positions[remainder]
            return f"{sign}{whole}.{prefix[:start]}({prefix[start:]})"
        return f"{sign}{whole}.{prefix}…"

    def convert(self, text: str, base_in: int, base_out: int, frac_digits: int | None = None) -> str:
        """Convert using an exact Fraction intermediary, preserving finite input exactly."""
        return self.from_decimal(self.parse_fraction(text, base_in), base_out, frac_digits)

    @staticmethod
    def _as_fraction(value: Any) -> Fraction:
        if isinstance(value, Fraction):
            return value
        if isinstance(value, int):
            return Fraction(value)
        if isinstance(value, mp.mpf):
            return Fraction(str(value))
        try:
            return Fraction(str(value))
        except (ValueError, ZeroDivisionError) as error:
            raise BaseConversionError(f"Значение {value!r} нельзя представить позиционно.") from error
