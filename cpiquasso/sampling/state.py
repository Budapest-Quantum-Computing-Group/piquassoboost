import piquasso as pq

from cpiquasso.sampling.backend import SamplingBackend


class SamplingState(pq.SamplingState):
    _backend_class = SamplingBackend
