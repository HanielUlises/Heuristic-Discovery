"""Repository layout and location of the compiled planner."""

from __future__ import annotations

import os
from pathlib import Path

_MARKERS = ("CMakeLists.txt", "cpp", "python")


def repository_root() -> Path:
    """Finds the repository root by walking up from this file."""
    for candidate in Path(__file__).resolve().parents:
        if all((candidate / marker).exists() for marker in _MARKERS):
            return candidate
    raise RuntimeError("cannot locate the repository root from " + str(Path(__file__).resolve()))


ROOT = repository_root()
INSTANCE_DIR = ROOT / "instances"
BENCHMARK_DIR = ROOT / "benchmarks"
RESULTS_DIR = ROOT / "results"

#: Directories searched for a compiled executable, relative to the repository root.
_BUILD_DIRS = (Path("build"), Path("build/Release"), Path("cmake-build-release"))


def _find_binary(name: str, env_var: str) -> Path:
    """Locates a compiled executable, honouring an environment override.

    ``env_var`` takes precedence over the search, which is what continuous
    integration and out-of-tree builds should set.
    """
    override = os.environ.get(env_var)
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"{env_var} points at a missing file: {path}")
        return path

    for directory in _BUILD_DIRS:
        for candidate in (ROOT / directory / name, ROOT / directory / f"{name}.exe"):
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"{name} was not found. Build it with:\n"
        "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\n"
        f"or set {env_var} to an existing executable."
    )


def planner_binary() -> Path:
    """Path of the ``hd_plan`` executable."""
    return _find_binary("hd_plan", "HD_PLANNER_BINARY")


def verifier_binary() -> Path:
    """Path of the ``hd_verify`` executable."""
    return _find_binary("hd_verify", "HD_VERIFIER_BINARY")


def relative_to_root(path: Path | str) -> str:
    """Path relative to the repository root when possible, absolute otherwise."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)
