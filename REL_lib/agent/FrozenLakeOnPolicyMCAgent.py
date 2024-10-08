import pickle
import numpy as np

from tqdm import tqdm
from typing import List, Tuple, Dict

from .GPIAgent import GPIAgent
from ..environment import CustomFrozenLakeENV


__all__ = ["FrozenOnPolicyMCAgent"]


class FrozenOnPolicyMCAgent(GPIAgent):
    def __init__(self,
                 env: CustomFrozenLakeENV,
                 epsilon: float = .2,
                 final_epsilon: float = .2,
                 decay_rate: float = 1e-2,
                 discount_factor: float = 0.5,
                 greedy_policy: bool = True,
                 eval_episodes: int = 1000,
                 impr_episodes: int = 1000,
                 max_trajectory_length: int = 100
                 ):
        super(FrozenOnPolicyMCAgent, self).__init__()
        self.__env = env
        self.__num_actions: int = self.__env.action_space.n
        self.__num_states: int = self.__env.observation_space.n

        self.__epsilon: float = epsilon
        self.__final_epsilon: float = final_epsilon
        self.__decay_rate: float = decay_rate

        self.__discount_factor: float = discount_factor
        self.__greedy_policy: bool = greedy_policy

        self.__eval_episodes: int = eval_episodes
        self.__impr_episodes: int = impr_episodes
        self.__max_trajectory_length: int = max_trajectory_length
    @staticmethod
    def _argmax_rand(state_q_values: np.ndarray) -> int:
        """
        :param q_values
        :return index of heighest value in values.
                Index is drawn by uniform distribution by numpy fn
        This works as np.argmax() but breaking ties randomly
        """
        return np.random.choice(np.flatnonzero(state_q_values == np.max(state_q_values)))

    def _save_weights(self, pi: np.ndarray) -> None:
        filename = f"{self.__class__.__name__}.pkl"
        obj = {
            "Metadata": {
                "discount_factor": self.__discount_factor,
                "greedy_policy": self.__greedy_policy,
                "eval_episodes": self.__eval_episodes,
                "impr_episodes": self.__impr_episodes
            },

            "pi": pi
        }

        with open(filename, "wb") as f:
            pickle.dump(obj, f)

    def _decay_epsilon(self) -> float:
        return max(self.__final_epsilon, self.__epsilon - self.__decay_rate)

    def _select_action(self, pi: np.ndarray[np.uint8], state: int) -> int:
        """
        Implementation both for epsilon-soft & epsilon-greedy policy
        :param pi: current policy for making decision
        :param state: current state of agent
        :return selected action
        """
        if np.random.rand() < self.__epsilon:
            action: int = self.__env.action_space.sample()
        else:
            action: int = np.argmax(pi[state, :])
        return action

    def __generate_trajectory(self,
                              pi: np.ndarray[np.float32],
                              max_len: int,
                              ) -> List[Tuple[int, int, float]]:
        """
        :param pi: current policy for making decision
        :return: generated trajectories
        """
        current_state: int = self.__env.env_start()

        trajectory: List[Tuple[int, int, float]] = []

        while True:
            current_action: int = self._select_action(pi, current_state)

            reward, next_state, term, _ = self.__env.env_step(current_action)
            trajectory.append((current_state, current_action, reward))

            current_state = next_state

            if term or len(trajectory) > max_len:
                break
        return trajectory

    def _eval_policy(self, pi: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        return_G: Dict[int, List[float]] = {state: [] for state in range(self.__num_states)}
        V: np.ndarray[np.float32] = np.random.random((self.__num_states, )).astype(np.float32)

        for _ in tqdm(range(1, self.__eval_episodes+1), "Evaluating policy", colour="red"):
            G = 0
            trajectory: List[Tuple[int, int, float]] = self.__generate_trajectory(pi, self.__max_trajectory_length)

            for t in range(len(trajectory)-2, -1, -1):
                G = self.__discount_factor * G + trajectory[t + 1][2]

                current_state, current_action = trajectory[t][:2]

                # First visit MC
                if current_state not in [trajectory[timestep][0] for timestep in range(0, t)]:
                    return_G[current_state].append(G)
                    V[current_state] = np.average(return_G[current_state])
        return V

    def _improve_policy(self, V: np.ndarray) -> np.ndarray[np.float32]:
        Q: np.ndarray[np.float32] = np.random.random((self.__num_states, self.__num_actions))
        pi: np.ndarray[np.float32] = np.ones((self.__num_states, self.__num_actions), dtype=np.float32) / self.__num_actions

        return_G: np.ndarray[np.float32] = np.zeros((self.__num_states, self.__num_actions), dtype=np.float32)
        SA_counter: np.ndarray[np.float32] = np.zeros((self.__num_states, self.__num_actions), dtype=np.uint32)

        for _ in tqdm(range(self.__impr_episodes), "Improve poliocy", colour="cyan"):
            G = 0
            trajectory: List[Tuple[int, int, float]] = self.__generate_trajectory(pi, self.__max_trajectory_length)

            for t in range(len(trajectory)-2, -1, -1):
                G = self.__discount_factor * G + trajectory[t + 1][2]

                St, At = trajectory[t][:2]

                # First visit MC
                if (St, At) not in [trajectory[timestep][:2] for timestep in range(0, t)]:
                    return_G[St, At] += G
                    SA_counter[St, At] += 1
                    Q[St, At] = return_G[St, At] / SA_counter[St, At]
                    optimal_A: int = self._argmax_rand(Q[St, :])

                    for action in range(self.__num_actions):
                        if action == optimal_A:
                            pi[St, action] = 1 - self.__epsilon * (1 + 1 / self.__num_actions)
                        else:
                            pi[St, action] = self.__epsilon / self.__num_actions
        return pi

    def train(self, num_iters: int = 100, save_weight: bool = True) -> np.ndarray[np.float32]:
        pi: np.ndarray[np.float32] = np.ones((self.__num_states, self.__num_actions), dtype=np.float32) / self.__num_actions

        for i in tqdm(range(num_iters), "Training in progress", colour="yellow"):
            new_V: np.ndarray[np.float32] = self._eval_policy(pi)
            new_pi: np.ndarray[np.uint8] = self._improve_policy(new_V)

            if np.array_equal(pi, new_pi):
                print(f"Policy converged at {i+1}-th iteration.\n")
                break

            pi = new_pi
            # self.__epsilon = self._decay_epsilon()

        if save_weight:
            self._save_weights(pi)
        return pi
