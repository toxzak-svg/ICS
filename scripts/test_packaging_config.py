"""Packaging configuration checks for console entry points."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_console_script_modules_are_included_in_package_discovery():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    finder = pyproject["tool"]["setuptools"]["packages"]["find"]
    include = set(finder.get("include", []))
    exclude = set(finder.get("exclude", []))

    for entry_point in scripts.values():
        module = entry_point.split(":", 1)[0]
        top_package = module.split(".", 1)[0]
        assert f"{top_package}*" in include
        assert f"{top_package}*" not in exclude


def main() -> int:
    tests = [test_console_script_modules_are_included_in_package_discovery]
    failures = []
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:
            print(f"FAIL {test.__name__}: {exc}")
            failures.append(test.__name__)
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print(f"ALL {len(tests)} TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
