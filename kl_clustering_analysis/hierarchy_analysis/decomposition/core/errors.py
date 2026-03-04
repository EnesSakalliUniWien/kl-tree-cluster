"""Typed exceptions for decomposition modules."""


class DecompositionValidationError(ValueError):
    """Raised when decomposition inputs or intermediate structures are invalid."""


class DecompositionMethodError(RuntimeError):
    """Raised when an unknown or unsupported decomposition method is requested."""

