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

from piquasso._simulators.gaussian.simulation_steps import (
    threshold_measurement as _pq_threshold_measurement,
)


def threshold_measurement(state, instruction, shots):
    """
    NOTE: This function calculates only by using torontonian.
    """
    # Temporary compatibility fallback for latest piquasso versions where the
    # boosted threshold path can abort in native code.
    return _pq_threshold_measurement(state, instruction, shots)
