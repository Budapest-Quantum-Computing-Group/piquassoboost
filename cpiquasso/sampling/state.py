import piquasso as pq

from cpiquasso.sampling.circuit import SamplingCircuit


class SamplingState(pq.SamplingState):
    _circuit_class = SamplingCircuit
