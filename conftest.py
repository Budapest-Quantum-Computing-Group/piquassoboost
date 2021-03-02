#
# ***REMOVED***
#

import pytest

from cpiquasso import patch


@pytest.fixture(autouse=True)
def _patch():
    patch()


