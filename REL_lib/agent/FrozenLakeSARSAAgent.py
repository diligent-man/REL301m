import pickle
import numpy as np

from tqdm import tqdm
from typing import List, Tuple, Dict

from .GPIAgent import GPIAgent
from ..environment import CustomFrozenLakeENV


__all__ = ["FrozenLakeSARSAAgent"]


class FrozenLakeSARSAAgent(GPIAgent):
    def __init__(self,
                 env: CustomFrozenLakeENV,
                 lr: float = 1e-3,
                 epsilon: float = 0.15,
                 discount_factor: float = 0.5,
                 eval_episodes: int = 1000,
                 impr_episodes: int = 1000
                 ):
        super(FrozenLakeSARSAAgent, self).__init__()
        self.__env = env
        self.__num_actions: int = self.__env.action_space.n
        self.__num_states: int = self.__env.observation_space.n

        self.__lr: float = lr
        self.__epsilon: float = epsilon
        self.__discount_factor: float = discount_factor

        self.__eval_episodes: int = eval_episodes
        self.__impr_episodes: int = impr_episodes

    def _save_weights(self, pi: np.ndarray) -> None:
        filename = f"{self.__class__.__name__}.pkl"
        obj = {
            "Metadata": {
                "lr": self.__lr,
                "epsilon": self.__epsilon,
                "discount_factor": self.__discount_factor,
                "eval_episodes": self.__eval_episodes,
                "impr_episodes": self.__impr_episodes
            },

            "pi": pi
        }

        with open(filename, "wb") as f:
            pickle.dump(obj, f)

    def _select_action(self, Q: np.ndarray[np.uint8], state: int) -> int:
        """
        :param Q: current state-action value
        :param state: current state of agent
        :return selected action
        """
        if np.random.rand() < self.__epsilon:
            action: int = self.__env.action_space.sample()
        else:
            action: int = np.random.choice(np.flatnonzero(Q[state] == np.max(Q[state, :])))
        return action

    def _eval_policy(self, Q: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        V: np.ndarray[np.float32] = np.random.random((self.__num_states, )).astype(np.float32)
        V[0] = V[self.__num_states-1] = 0  # Terminal states are 0

        for _ in tqdm(range(self.__eval_episodes), "Evaluating policy", colour="red"):
            St: int = self.__env.env_start()

            while True:
                At: int = np.argmax(Q[St, :])
                reward, next_state, term, _ = self.__env.step(At)

                TD_err: float = reward + self.__discount_factor * V[next_state] - V[St]
                V[St] += self.__lr * TD_err

                if term:
                    break
                St = next_state
        return V

    def _improve_policy(self, V: np.ndarray) -> np.ndarray[np.float32]:
        Q: np.ndarray[np.float32] = np.random.standard_normal((self.__num_states, self.__num_actions))
        Q[0, :] = Q[self.__num_states-1, :] = 0
        pi: np.ndarray[np.uint8] = np.random.randint(low=0, high=self.__num_actions, size=(self.__num_states, ))

        for _ in tqdm(range(self.__impr_episodes), "Improving policy", colour="cyan"):
            St: int = self.__env.env_start()
            At: int = self._select_action(Q, St)

            while True:
                reward, next_state, term, _ = self.__env.step(At)
                next_action: int = self._select_action(Q, next_state)

                SARSA_err = reward + self.__discount_factor * Q[next_state, next_action] - Q[St, At]
                Q[St, At] += self.__lr * SARSA_err

                if term:
                    break

                St = next_state
                At = next_action
        return Q

    def train(self, num_iters: int = 100, save_weight: bool = True) -> np.ndarray[np.float32]:
        Q: np.ndarray[np.float32] = np.random.standard_normal((self.__num_states, self.__num_actions))

        for i in tqdm(range(num_iters), "Training in progress", colour="yellow"):
            new_V: np.ndarray[np.float32] = self._eval_policy(Q)
            new_Q: np.ndarray[np.uint8] = self._improve_policy(new_V)

            if np.array_equal(Q, new_Q):
                print(f"Policy converged at {i+1}-th iteration.\n")
                break

            Q = new_Q

        if save_weight:
            self._save_weights(Q)
        return Q
