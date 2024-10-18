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

__version__ = "0.2.0"
