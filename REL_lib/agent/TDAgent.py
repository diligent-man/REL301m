import numpy as np

from typing import Dict
from .BaseAgent import BaseAgent
from ..environment import _reward, _obs

__all__ = ["TDAgent"]


class TDAgent(BaseAgent):
    __rand_generator: np.random.RandomState
    __discount: float
    __step_size: float
    __last_state: int | None
    __values: np.ndarray  # shape: (# states, )
    __policy: np.ndarray  # shape: (# states, # actions)

    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return f"""Discount: {self.__discount}
Step size: {self.__step_size}
Last state: {self.__last_state}
Policy : {self.__policy}
Values : {self.__values}
"""

    @property
    def policy(self) -> np.ndarray:
        return self.__policy

    @property
    def values(self) -> np.ndarray:
        return self.__values

    def agent_init(self, agent_info: Dict | None = None) -> None:
        if agent_info is None:
            agent_info = {}

        assert "policy" in agent_info.keys()
        assert "values" in agent_info.keys()

        self.__rand_generator = np.random.RandomState(seed=agent_info.get("seed", 9999))
        self.__discount = agent_info.get("discount", .5)
        self.__step_size = agent_info.get("step_size", 1e-3)

        self.__values = agent_info["values"]
        self.__policy = agent_info["policy"]

    def agent_start(self, obs: _obs) -> int:
        """
        :param obs: the state from the environment's env_start function.
        :return: 1st action the agent takes
        """
        num_actions: int = self.__policy.shape[1]
        corresponding_prob: float = self.policy[obs]

        action = self.__rand_generator.choice(range(num_actions), p=corresponding_prob)
        self.__last_state = obs
        return action

    def agent_step(self, reward: _reward, obs: _obs) -> int:
        """
        :param reward: the reward received by the agent.
        :param obs: the state from the environment's env_start function.'
        :return: The action the agent is taking.
        """
        # Tabular TD update rule: V[S_t] = V[S_t] + alpha * (Gt - V[S_t])
        Gt: float = reward + self.__discount * self.__values[obs]
        self.__values[self.__last_state] += self.__step_size * (Gt - self.__values[self.__last_state])

        action: int = self.__rand_generator.choice(range(self.__policy.shape[1]), p=self.__policy[obs])
        self.__last_state = obs
        return action

    def agent_end(self, reward: int) -> None:
        """
        Run when the agent terminates.
        :param reward: the reward the agent received for entering the terminal state.
        """
        self.__values[self.__last_state] += self.__step_size * (reward - self.__values[self.__last_state])

    def agent_cleanup(self) -> None:
        """Cleanup done after the agent ends."""
        self.__last_state = None

    def agent_message(self, message: str) -> np.ndarray:
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """
        if message == "get_values":
            return self.__values
        elif message == "get_policy":
            return self.__policy
        else:
            raise Exception("TDAgent.agent_message(): Message not understood!")
