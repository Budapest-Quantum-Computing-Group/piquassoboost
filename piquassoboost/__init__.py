#
# Copyright 2021-2022 Budapest Quantum Computing Group
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

import piquasso as pq

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
