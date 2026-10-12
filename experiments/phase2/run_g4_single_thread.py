"""Freeze thread environment before importing diagnostic NumPy/flopscope."""

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

from capture_g4_features import main  # noqa: E402

if __name__ == "__main__":
    main()
