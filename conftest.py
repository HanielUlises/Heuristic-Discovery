"""Makes the ``hd`` package importable from the source tree without installing it."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))
