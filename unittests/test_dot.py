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

import numpy as np

from scipy.stats import unitary_group

import piquassoboost.common.dot_wrapper as dot


def test_dot2():
    A_numpy = np.ascontiguousarray(unitary_group.rvs(2))
    B_numpy = np.ascontiguousarray(unitary_group.rvs(2))

    print(A_numpy)
    print(B_numpy.conjugate())

    C = dot.dot2(A_numpy, B_numpy)

    C_numpy = A_numpy @ B_numpy.conjugate()

    assert np.allclose(C_numpy, C)
