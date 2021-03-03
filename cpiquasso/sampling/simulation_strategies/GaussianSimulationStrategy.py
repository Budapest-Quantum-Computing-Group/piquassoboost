


from .GaussianSimulationStrategy_wrapper import (
    GaussianSimulationStrategy_wrapper
)


class GaussianSimulationStrategy(
    GaussianSimulationStrategy_wrapper
):
    def __init__(self, covariance_matrix):

        super().__init__(covariance_matrix)


    def simulate(self, samples_number: int = 1):
        """
            Returns sample from linear optics experiments given output state.
            :param input_state: Input state in particle basis.
            :return: A resultant state after traversing through interferometer.
        """

        return super().simulate(samples_number)
