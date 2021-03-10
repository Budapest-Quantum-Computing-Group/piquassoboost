


from .GaussianSimulationStrategy_wrapper import (
    GaussianSimulationStrategy_wrapper
)


class GaussianSimulationStrategy(
    GaussianSimulationStrategy_wrapper
):
    def __init__(self, covariance_matrix, m=None, fock_cutoff=5, max_photons=20):

        super().__init__(covariance_matrix=covariance_matrix, m=m, fock_cutoff=fock_cutoff, max_photons=max_photons)


    def simulate(self, samples_number: int = 1):
        """
            Returns sample from linear optics experiments given output state.
            :param input_state: Input state in particle basis.
            :return: A resultant state after traversing through interferometer.
        """

        return super().simulate(samples_number)
