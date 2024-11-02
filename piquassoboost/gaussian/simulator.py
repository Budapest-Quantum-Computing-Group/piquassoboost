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

import piquasso as pq

from piquasso.instructions import measurements

from piquassoboost.config import BoostConfig
from piquassoboost.connector import BoostConnector

from .calculations import threshold_measurement

class BoostedGaussianSimulator(pq.GaussianSimulator):
    _instruction_map = {
        **pq.GaussianSimulator._instruction_map,
        measurements.ThresholdMeasurement: threshold_measurement,
    }

    _config_class = BoostConfig
    _connector_class = BoostConnector
