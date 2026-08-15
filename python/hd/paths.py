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

_BINARY_CANDIDATES = (
    Path("build/hd_plan"),
    Path("build/Release/hd_plan"),
    Path("cmake-build-release/hd_plan"),
    Path("build/hd_plan.exe"),
)


def planner_binary() -> Path:
    """Path of the ``hd_plan`` executable.

    ``HD_PLANNER_BINARY`` overrides the search, which is what continuous
    integration and out-of-tree builds should set.
    """
    override = os.environ.get("HD_PLANNER_BINARY")
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"HD_PLANNER_BINARY points at a missing file: {path}")
        return path

    for relative in _BINARY_CANDIDATES:
        candidate = ROOT / relative
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "hd_plan was not found. Build it with:\n"
        "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\n"
        "or set HD_PLANNER_BINARY to an existing executable."
    )


def relative_to_root(path: Path | str) -> str:
    """Path relative to the repository root when possible, absolute otherwise."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)
