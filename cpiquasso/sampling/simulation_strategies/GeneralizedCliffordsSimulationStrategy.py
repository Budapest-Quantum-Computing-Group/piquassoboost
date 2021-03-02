


from .GeneralizedCliffordsSimulationStrategy_wrapper import (
    GeneralizedCliffordsSimulationStrategy_wrapper
)


class GeneralizedCliffordsSimulationStrategy(
    GeneralizedCliffordsSimulationStrategy_wrapper
):
    def __init__(self, interferometer_matrix):

        super().__init__(interferometer_matrix)


    def simulate(self, input_state, samples_number: int = 1):
        """
            Returns sample from linear optics experiments given output state.
            :param input_state: Input state in particle basis.
            :return: A resultant state after traversing through interferometer.
        """

        return super().simulate(input_state, samples_number)
