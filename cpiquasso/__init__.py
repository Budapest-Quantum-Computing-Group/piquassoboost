
import piquasso as pq

from cpiquasso.gaussian.state import GaussianState
from cpiquasso.sampling.state import SamplingState


def patch():
    class CPiquassoPlugin:
        classes = {
            "GaussianState": GaussianState,
            "SamplingState": SamplingState,
        }

    pq.use(CPiquassoPlugin)