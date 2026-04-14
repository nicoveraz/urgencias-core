"""Trivial smoke test: confirms the package imports and pytest works."""

import urgencias_core


def test_package_imports():
    assert urgencias_core.__doc__
