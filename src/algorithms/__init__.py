"""Algorithm factory and registration module."""

from src.algorithms.factory import ALGORITHM_REGISTRY, build_algorithm

__all__ = ["ALGORITHM_REGISTRY", "build_algorithm"]
