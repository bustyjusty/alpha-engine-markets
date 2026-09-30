"""Tests for the publish preflight checks.

The dependency check is the one worth testing hard: it exists because an
import that resolves from the local virtual environment but is missing from
requirements.txt passes every other check, works perfectly on this machine,
and then breaks the deployed site. A false negative here ships a broken build.
"""

from __future__ import annotations

import pathlib

import pytest

from scripts.preflight import (
    declared_packages,
    top_level_imports,
    undeclared_imports,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent


# --- Parsing requirements.txt -------------------------------------------------


def test_declared_packages_strips_version_specifiers() -> None:
    assert declared_packages("pandas>=2.2\nyfinance>=1.5,<2\nanthropic") == {
        "pandas",
        "yfinance",
        "anthropic",
    }


def test_declared_packages_ignores_comments_and_blanks() -> None:
    text = "# a comment\n\npandas>=2.2  # trailing note\n-r other.txt\n"
    assert declared_packages(text) == {"pandas"}


def test_declared_packages_handles_extras() -> None:
    assert declared_packages("uvicorn[standard]>=0.30") == {"uvicorn"}


# --- Parsing imports ----------------------------------------------------------


def test_top_level_imports_finds_both_import_forms(tmp_path) -> None:
    source = tmp_path / "module.py"
    source.write_text(
        "import pandas as pd\n"
        "import plotly.graph_objects as go\n"
        "from yfinance import Ticker\n"
        "from market_intel.config import Settings\n",
        encoding="utf-8",
    )
    assert top_level_imports([source]) == {
        "pandas",
        "plotly",
        "yfinance",
        "market_intel",
    }


def test_relative_imports_are_not_dependencies(tmp_path) -> None:
    source = tmp_path / "module.py"
    source.write_text("from . import sibling\nfrom .. import parent\n", encoding="utf-8")
    assert top_level_imports([source]) == set()


def test_a_module_named_in_a_string_is_not_an_import(tmp_path) -> None:
    """Parsed, not grepped — a docstring mentioning scipy is not a dependency."""
    source = tmp_path / "module.py"
    source.write_text('"""Unlike scipy, this uses import numpy only in docs."""\n', encoding="utf-8")
    assert top_level_imports([source]) == set()


def test_a_syntax_error_does_not_crash_the_check(tmp_path) -> None:
    """The test run reports broken syntax; preflight must not blow up first."""
    source = tmp_path / "broken.py"
    source.write_text("def oops(\n", encoding="utf-8")
    assert top_level_imports([source]) == set()


# --- The check itself ---------------------------------------------------------


def test_an_undeclared_third_party_import_is_caught() -> None:
    missing = undeclared_imports({"pandas", "scipy"}, {"pandas"})
    assert missing == {"scipy"}


def test_standard_library_imports_are_never_required() -> None:
    assert undeclared_imports({"datetime", "json", "pathlib", "math"}, set()) == set()


def test_first_party_packages_are_never_required() -> None:
    assert undeclared_imports({"market_intel", "ui", "tests", "scripts"}, set()) == set()


def test_an_import_whose_distribution_is_named_differently_is_accepted() -> None:
    """`import pydantic_settings` is satisfied by `pydantic-settings`."""
    assert undeclared_imports({"pydantic_settings"}, {"pydantic-settings"}) == set()


def test_underscore_and_hyphen_spellings_both_match() -> None:
    assert undeclared_imports({"some_pkg"}, {"some-pkg"}) == set()


# --- The real project ---------------------------------------------------------


def test_the_deployed_app_declares_every_package_it_imports() -> None:
    """The check that would have caught numpy missing from requirements.txt."""
    files = [ROOT / "app.py"]
    for tree in ("market_intel", "ui"):
        files.extend(
            path
            for path in (ROOT / tree).rglob("*.py")
            if "__pycache__" not in path.parts
        )
    declared = declared_packages((ROOT / "requirements.txt").read_text(encoding="utf-8"))
    missing = undeclared_imports(top_level_imports(files), declared)
    assert not missing, (
        f"imported but not in requirements.txt: {sorted(missing)} — "
        "this works locally and fails on Streamlit Cloud"
    )


def test_the_env_file_is_not_tracked() -> None:
    """A tracked .env would publish the API keys to a public repository."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", ".env"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    assert tracked == "", ".env is tracked by git — it must stay ignored"
