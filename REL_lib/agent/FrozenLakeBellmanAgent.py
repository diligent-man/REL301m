import numpy as np

from tqdm import tqdm
from .GPIAgent import GPIAgent
from ..environment import CustomFrozenLakeENV


__all__ = ["FrozenBellmanAgent"]


class FrozenBellmanAgent(GPIAgent):
    def __init__(self,
                 env: CustomFrozenLakeENV,
                 threshold: float = 1e-10,
                 discount_factor: float = 0.5,
                 ):
        super().__init__()
        self.__env = env
        self.__threshold: float = threshold
        self.__discount_factor: float = discount_factor
        self.__num_actions: int = self.__env.action_space.n
        self.__num_states: int = self.__env.observation_space.n

    def _eval_policy(self, pi: np.ndarray) -> np.ndarray:
        updated_V: np.ndarray = np.zeros(self.__num_states)

        while True:
            current_V = updated_V.copy()
            for state in range(self.__num_states):
                action = pi[state]
                updated_V[state] = sum([prob * (reward + self.__discount_factor * current_V[s_prime]) \
                                        for (prob, s_prime, reward, _) in self.__env.P[state][action]
                                        ])

            if np.sum((np.fabs(updated_V - current_V))) <= self.__threshold:
                break
        return updated_V

    def _improve_policy(self, V: np.ndarray) -> np.ndarray:
        policy = np.zeros(self.__num_states)

        for state in range(self.__num_states):
            state_q_value = np.zeros(self.__num_actions)
            for action in range(self.__num_actions):
                state_q_value[action] = sum([prob * (reward + self.__discount_factor * V[s_prime]) \
                                             for (prob, s_prime, reward, _) in self.__env.P[state][action]
                                             ])
            policy[state] = np.argmax(state_q_value)
        return policy

    def train(self, num_iters: int = 2000000) -> np.ndarray:
        """
        :param num_iters: number of iterations to train agent
        :return trained policy
        """
        pi = np.zeros(self.__num_states)

        for i in tqdm(range(num_iters), "Training in progress"):
            new_V: np.ndarray = self._eval_policy(pi)
            new_pi = self._improve_policy(new_V)

            if np.array_equal(pi, new_pi):
                print(f"Policy-Iteration converged at step {i+1}.\n")
                break
            pi = new_pi
        return pi
