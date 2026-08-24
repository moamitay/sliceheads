"""Packaging test: assert tensorflow and tsai are not importable."""

import importlib
import pytest


@pytest.mark.parametrize("forbidden", ["tensorflow", "tsai"])
def test_forbidden_dependency_not_importable(forbidden):
    try:
        importlib.import_module(forbidden)
        pytest.fail(f"Forbidden dependency '{forbidden}' is importable — remove it from the dependency tree.")
    except ImportError:
        pass  # expected
