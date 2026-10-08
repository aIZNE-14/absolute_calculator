from __future__ import annotations


class UMBACError(ValueError):
    """Base calculation error with a user-facing hint."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


class BaseConversionError(UMBACError):
    """Invalid base, digit, or conversion request."""


class ParseError(UMBACError):
    """Invalid or unsupported mathematical expression."""


class DivergenceError(UMBACError):
    """An improper integral or limit does not converge to a finite result."""


class DomainError(UMBACError):
    """An operation is outside its mathematical domain."""


class SolverError(UMBACError):
    """A requested differential or physical problem is unsupported or unsolved."""


class PhysicsError(UMBACError):
    """Physical parameters or state invariants are invalid."""


class TimeoutError_(UMBACError):
    """A calculation exceeded its configured time limit."""
