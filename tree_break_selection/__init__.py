"""Tree-Break Selection hierarchy decomposition package."""

from importlib import import_module

__version__ = "0.1.0"

__all__ = [
    "hierarchy_analysis",
    "plot",
    "space_separation",
    "tree",
]


def __getattr__(name: str):
    """Load public subpackages only when a caller requests them."""
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
