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

import os
import sys
import importlib

# On Linux, libtbbmalloc_proxy.so.2 must be loaded globally *before* any C++
# extension imports so that its malloc/free override reaches all loaded DSOs
# (including libstdc++.so.6 which was already loaded by the Python runtime).
# Without RTLD_GLOBAL the proxy loads as an RTLD_LOCAL dependency and cannot
# intercept allocations made by libstdc++ before the proxy came in, causing
# "free(): invalid pointer" aborts when unordered_map or vector rehash is used.
if sys.platform != "win32":
    import ctypes as _ctypes
    import subprocess as _subprocess
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    # Build candidate paths.  Priority:
    # 1. Same dir as libpiquassoboost.so (resolved via ldd to find the exact TBB build)
    # 2. Package directory (for installed wheel bundles)
    # 3. System path (ldconfig fallback)
    _proxy_candidates = []
    _libpb = os.path.join(_pkg_dir, "libpiquassoboost.so.0.1")
    if os.path.exists(_libpb):
        try:
            _ldd_out = _subprocess.check_output(
                ["ldd", _libpb], stderr=_subprocess.DEVNULL, text=True
            )
            for _line in _ldd_out.splitlines():
                if "libtbbmalloc_proxy" in _line and "=>" in _line:
                    _ldd_path = _line.split("=>")[1].strip().split()[0]
                    if os.path.exists(_ldd_path):
                        _proxy_candidates.append(_ldd_path)
                    break
        except Exception:
            pass
    _proxy_candidates.append(os.path.join(_pkg_dir, "libtbbmalloc_proxy.so.2"))
    _proxy_candidates.append("libtbbmalloc_proxy.so.2")
    for _proxy_path in _proxy_candidates:
        try:
            _ctypes.CDLL(_proxy_path, mode=_ctypes.RTLD_GLOBAL)
            break
        except OSError:
            continue
    del _ctypes, _subprocess, _pkg_dir, _libpb, _proxy_candidates, _proxy_path

# On Windows, Python 3.8+ no longer searches PATH for DLLs loaded by
# extension modules.  Add the package directory explicitly so that
# piquassoboost.dll (built alongside the .pyd files) is found at import time.
# Also add the BLAS/TBB runtime DLL directory (conda Library/bin or BLAS_LIB_DIR/../bin).
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    os.add_dll_directory(_pkg_dir)
    # Try BLAS_LIB_DIR env var (set during build); resolve ../bin as runtime sibling
    _blas_lib_dir = os.environ.get("BLAS_LIB_DIR", "")
    if _blas_lib_dir:
        _blas_bin_dir = os.path.join(_blas_lib_dir, "..", "bin")
        if os.path.isdir(_blas_bin_dir):
            os.add_dll_directory(os.path.normpath(_blas_bin_dir))
        elif os.path.isdir(_blas_lib_dir):
            os.add_dll_directory(_blas_lib_dir)
    # Also try CONDA_PREFIX/Library/bin
    _conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if _conda_prefix:
        _conda_bin = os.path.join(_conda_prefix, "bin")
        if os.path.isdir(_conda_bin):
            os.add_dll_directory(_conda_bin)

def _import_piquasso():
    try:
        import piquasso as _pq

        return _pq
    except ModuleNotFoundError as exc:
        # When running from this repository, `piquasso` is often a source symlink
        # without compiled `_math` extensions. In that case, prefer the installed
        # package distribution that ships/builds those native modules.
        if exc.name != "piquasso._math.permanent":
            raise

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        removed_paths = []

        for path_entry in list(sys.path):
            normalized_entry = os.path.abspath(path_entry or os.getcwd())
            local_piquasso = os.path.join(normalized_entry, "piquasso")

            if os.path.isdir(local_piquasso) and os.path.realpath(local_piquasso).startswith(
                os.path.realpath(repo_root)
            ):
                removed_paths.append(path_entry)
                sys.path.remove(path_entry)

        importlib.invalidate_caches()

        try:
            import piquasso as _pq

            return _pq
        finally:
            for path_entry in reversed(removed_paths):
                sys.path.insert(0, path_entry)


pq = _import_piquasso()

from piquassoboost.config import BoostConfig
from piquassoboost.connector import BoostConnector

from piquassoboost.gaussian.simulator import BoostedGaussianSimulator
from piquassoboost.sampling.simulator import BoostedSamplingSimulator
from piquassoboost.fock.pure.simulator import BoostedPureFockSimulator
from piquassoboost.fock.general.simulator import BoostedFockSimulator


def patch():
    pq.BoostedGaussianSimulator = BoostedGaussianSimulator
    pq.BoostedSamplingSimulator = BoostedSamplingSimulator
    pq.BoostedPureFockSimulator = BoostedPureFockSimulator
    pq.BoostedFockSimulator = BoostedFockSimulator

    pq.BoostConfig = BoostConfig
    pq.BoostConnector = BoostConnector

    pq.GaussianSimulator = BoostedGaussianSimulator
    pq.SamplingSimulator = BoostedSamplingSimulator
    pq.PureFockSimulator = BoostedPureFockSimulator
    pq.FockSimulator = BoostedFockSimulator

    pq.Config = BoostConfig

__version__ = "0.3.1"
