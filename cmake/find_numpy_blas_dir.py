#!/usr/bin/env python3
#
# Copyright 2021-2026 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utility script invoked from CMake to locate the BLAS directory used by NumPy.

Tries newer NumPy APIs first, then falls back to older mechanisms, finally
defaulting to the standard ../lib location inside the NumPy installation.
Prints the detected directory to stdout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _try_new_api(numpy_module) -> "str | None":
    """Attempt to locate BLAS directory using newer NumPy internals."""
    try:
        from numpy._core._multiarray_umath import __cpu_features__  # noqa: F401
    except Exception:
        return None

    numpy_dir = Path(numpy_module.__file__).resolve().parent
    candidates = [
        numpy_dir / ".." / "lib",
        numpy_dir / ".." / ".." / "lib",
        Path("/usr/lib"),
        Path("/usr/local/lib"),
    ]

    for path in candidates:
        abs_path = path.resolve()
        if abs_path.exists():
            return str(abs_path)
    return None


def _try_config_api(numpy_module) -> "str | None":
    """Fallback to NumPy's configuration metadata."""
    try:
        blas_info = numpy_module.__config__.get_info("blas_opt_info")
    except Exception:
        return None

    libs = blas_info.get("library_dirs", []) if isinstance(blas_info, dict) else []
    if libs:
        return os.path.abspath(libs[0])
    return None


def _default_path(numpy_module) -> str:
    """Final fallback: assume BLAS resides in ../lib next to NumPy."""
    numpy_dir = Path(numpy_module.__file__).resolve().parent
    return str((numpy_dir / ".." / "lib").resolve())


def main() -> int:
    try:
        import numpy  # type: ignore
    except Exception as exc:
        print(f"Failed to import numpy: {exc}", file=sys.stderr)
        return 1

    for resolver in (_try_new_api, _try_config_api):
        try:
            result = resolver(numpy)
        except Exception:
            result = None
        if result:
            print(result)
            return 0

    # Last resort
    print(_default_path(numpy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
