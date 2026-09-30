"""Checks that run before a publish, so a bad push never reaches the live site.

Pushing is deploying: Streamlit Cloud rebuilds from whatever lands on `main`,
and a broken build is visible to anyone holding the link. `git add -A` already
sweeps every file, so "I forgot to add it" is not the real risk. These four
are:

1. **Red tests.** The site rebuilds regardless of whether the code works.
2. **An undeclared dependency.** A new `import` works locally because the
   package is in the venv, then fails on Cloud because it is not in
   requirements.txt. This is the one that bit us with numpy.
3. **A secret.** `.env` is ignored, but a key pasted into tracked source
   would publish to a public repo, and rotating it is the only fix.
4. **A needed file that is ignored.** Anything under the source tree matching
   .gitignore silently stays on this machine.

Run directly (``python scripts/preflight.py``) or through publish.bat, which
refuses to push when this exits non-zero. ``--skip-tests`` is for a
documentation-only change where the suite has already run.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Source trees whose imports must be covered by requirements.txt. Tests are
#: excluded: they run here, never on the deployed app.
SOURCE_PATHS = ("market_intel", "ui", "app.py")

#: Import name -> distribution name, where they differ. Only needed for the
#: handful of packages that do not name their module after themselves.
IMPORT_ALIASES = {
    "pydantic_settings": "pydantic-settings",
    "dateutil": "python-dateutil",
    "yaml": "PyYAML",
    "dotenv": "python-dotenv",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4",
}

#: Things that look like a leaked credential in tracked source.
SECRET_PATTERNS = (
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"), "Anthropic API key"),
    (re.compile(r"\bghp_[A-Za-z0-9]{36}\b"), "GitHub personal access token"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key id"),
    # Assignment of a long opaque literal to something named like a key.
    (
        re.compile(
            r"(?i)\b(api[_-]?key|secret|token|password)\b\s*[:=]\s*"
            r"['\"][A-Za-z0-9_\-]{16,}['\"]"
        ),
        "hardcoded credential",
    ),
)

#: Ignored files that legitimately live in the source tree.
IGNORED_OK = (".pyc", ".pyo")
IGNORED_OK_DIRS = ("__pycache__", ".pytest_cache", ".egg-info")


class Result:
    """Accumulates findings so every check runs before anything is reported."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def fail(self, check: str, detail: str) -> None:
        self.failures.append(f"{check}: {detail}")

    def note(self, detail: str) -> None:
        self.notes.append(detail)

    @property
    def ok(self) -> bool:
        return not self.failures


def _git(*args: str) -> str:
    """Run a git command in the repo and return stdout."""
    done = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return done.stdout


def _python_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for entry in SOURCE_PATHS:
        path = ROOT / entry
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                item
                for item in path.rglob("*.py")
                if "__pycache__" not in item.parts
            )
    return files


def top_level_imports(files: list[pathlib.Path]) -> set[str]:
    """Every top-level module name imported by the given files.

    Parsed rather than grepped, so a module named inside a string or comment
    is not mistaken for a dependency.
    """
    found: set[str] = set()
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue  # reported by the test run, not here
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def declared_packages(requirements: str) -> set[str]:
    """Distribution names listed in requirements.txt, lowercased."""
    names: set[str] = set()
    for line in requirements.splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~\[ ]", line, maxsplit=1)[0].strip()
        if name:
            names.add(name.lower())
    return names


def undeclared_imports(imports: set[str], declared: set[str]) -> set[str]:
    """Third-party imports with no matching line in requirements.txt."""
    local = {"market_intel", "ui", "tests", "scripts"}
    missing: set[str] = set()
    for name in sorted(imports):
        if name in sys.stdlib_module_names or name in local or name.startswith("_"):
            continue
        distribution = IMPORT_ALIASES.get(name, name).lower()
        if distribution not in declared and distribution.replace("_", "-") not in declared:
            missing.add(name)
    return missing


def check_dependencies(result: Result) -> None:
    """Every import the deployed app makes must be installable on Cloud."""
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    missing = undeclared_imports(
        top_level_imports(_python_files()), declared_packages(requirements)
    )
    if missing:
        result.fail(
            "dependencies",
            f"imported but not in requirements.txt: {', '.join(sorted(missing))}. "
            "This works here and fails on Streamlit Cloud.",
        )


def check_secrets(result: Result) -> None:
    """Nothing credential-shaped may enter a public repo."""
    tracked = [
        line
        for line in _git("ls-files").splitlines()
        if line.endswith((".py", ".toml", ".md", ".txt", ".bat", ".json", ".yaml"))
    ]
    for name in tracked:
        path = ROOT / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern, label in SECRET_PATTERNS:
            match = pattern.search(text)
            if match:
                line_no = text[: match.start()].count("\n") + 1
                result.fail("secrets", f"possible {label} in {name}:{line_no}")

    if ".env" in _git("ls-files").splitlines():
        result.fail("secrets", ".env is tracked by git; it must stay ignored")


def check_ignored_source(result: Result) -> None:
    """A source file matching .gitignore would never reach the site."""
    output = _git("status", "--ignored=matching", "--short")
    for line in output.splitlines():
        if not line.startswith("!!"):
            continue
        name = line[3:].strip()
        if not name.startswith(("market_intel/", "ui/", "tests/", "scripts/")):
            continue
        if name.endswith(IGNORED_OK) or any(part in name for part in IGNORED_OK_DIRS):
            continue
        result.fail(
            "ignored files",
            f"{name} is inside the source tree but gitignored; it will not ship",
        )


def check_tests(result: Result) -> None:
    """The live site rebuilds whether or not the code works."""
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    tail = [line for line in done.stdout.strip().splitlines() if line][-1:]
    if done.returncode != 0:
        result.fail("tests", tail[0] if tail else "pytest failed")
    else:
        result.note(tail[0] if tail else "tests passed")


def summarise_changes(result: Result) -> None:
    """Say what is about to ship, so a surprise is caught before the push."""
    changes = [line for line in _git("status", "--short").splitlines() if line.strip()]
    if not changes:
        result.note("no local changes; nothing to publish")
        return
    result.note(f"{len(changes)} files to publish")
    for line in changes[:20]:
        result.note(f"    {line}")
    if len(changes) > 20:
        result.note(f"    ... and {len(changes) - 20} more")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="skip the suite (documentation-only changes)",
    )
    args = parser.parse_args()

    result = Result()
    check_dependencies(result)
    check_secrets(result)
    check_ignored_source(result)
    if not args.skip_tests:
        check_tests(result)
    summarise_changes(result)

    for note in result.notes:
        print(note)
    if result.ok:
        print("\nPreflight passed.")
        return 0
    print("\nPreflight FAILED - nothing was pushed:")
    for failure in result.failures:
        print(f"  - {failure}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
