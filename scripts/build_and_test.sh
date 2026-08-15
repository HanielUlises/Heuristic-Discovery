#!/usr/bin/env bash
# Configures and builds the engine, then runs both test suites.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_type="${1:-Release}"

cmake -S "$root" -B "$root/build" -DCMAKE_BUILD_TYPE="$build_type"
cmake --build "$root/build" -j

"$root/build/cpp/tests/hd_tests"
PYTHONPATH="$root/python" python3 -m pytest "$root" -q
