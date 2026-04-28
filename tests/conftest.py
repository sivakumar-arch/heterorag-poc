# tests/conftest.py
"""
Pytest configuration for HeteroRAG test suite.

Markers:
    integration  — requires live Docker services (postgres, neo4j, elasticsearch).
                   Skipped automatically when services are not reachable.
                   Run explicitly with: pytest -m integration

Usage:
    pytest                              # unit tests only (fast, no Docker needed)
    pytest -m integration               # integration tests only
    pytest tests/layer1/ -v             # all layer 1 tests (unit + integration)
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require live Docker Compose services. "
        "Each test class uses its own skip guard based on actual service reachability.",
    )


def pytest_collection_modifyitems(config, items):
    """
    If --no-integration flag is passed, skip all integration-marked tests.
    Default behaviour: integration tests run but skip themselves if unreachable.
    """
    if config.getoption("--no-integration", default=False):
        skip_marker = pytest.mark.skip(reason="--no-integration flag set")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_marker)


def pytest_addoption(parser):
    parser.addoption(
        "--no-integration",
        action="store_true",
        default=False,
        help="Skip all integration tests (useful in CI without Docker).",
    )
