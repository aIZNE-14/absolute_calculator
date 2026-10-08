from fractions import Fraction

import pytest

from umbac.base_converter import ALPHABET, BaseConverter
from umbac.errors import BaseConversionError


def test_reference_integer_conversions():
    converter = BaseConverter()
    assert converter.convert("255", 10, 16) == "FF"
    assert converter.convert("FF", 16, 2) == "11111111"
    assert BaseConverter.parse_fraction("Z", 64) == 35
    assert BaseConverter.parse_fraction("z", 64) == 61
    assert BaseConverter.parse_fraction("ff", 16) == 255


def test_full_base64_alphabet_round_trip():
    converter = BaseConverter()
    assert len(ALPHABET) == 64
    for base in range(2, 65):
        for value in (0, 1, base - 1, base, base ** 2 + 17):
            encoded = converter.from_decimal(value, base)
            assert BaseConverter.parse_fraction(encoded, base) == value


def test_periodic_fraction_has_repeating_parentheses():
    converter = BaseConverter()
    assert converter.from_decimal_periodic(Fraction(1, 10), 2) == "0.0(0011)"


def test_decimal_expression_rounding_carries_into_integer():
    assert BaseConverter().from_decimal(Fraction(7, 4), 2, frac_digits=1) == "10"


@pytest.mark.parametrize("text,base", [("2", 2), ("/", 63), ("1.2", 2), ("1..0", 10), ("", 10)])
def test_invalid_digits_and_formats_raise_positioned_error(text, base):
    with pytest.raises(BaseConversionError):
        BaseConverter.parse_fraction(text, base)


def test_base_outside_supported_range_is_rejected():
    with pytest.raises(BaseConversionError):
        BaseConverter.validate_base(65)
