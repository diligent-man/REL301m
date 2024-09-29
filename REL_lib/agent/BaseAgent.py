import numpy as np


from typing import Dict, List
from abc import ABCMeta, abstractmethod
from ..environment import _reward, _obs

__all__ = ["BaseAgent"]


class BaseAgent(metaclass=ABCMeta):
    """Implements the agent for an RL-Glue environment."""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def agent_init(self, agent_info: Dict | None):
        """Setup for the agent called when the experiment first starts."""
        raise NotImplementedError

    @abstractmethod
    def agent_start(self, obs: _obs):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            obs (Numpy array): the state observation from the environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        raise NotImplementedError

    @abstractmethod
    def agent_step(self, reward: _reward, obs: _obs):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            obs (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        raise NotImplementedError

    @abstractmethod
    def agent_end(self, reward: _reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the terminal state.
        """
        raise NotImplementedError

    @abstractmethod
    def agent_cleanup(self):
        """Cleanup done after the agent ends."""
        raise NotImplementedError

    @abstractmethod
    def agent_message(self, message: str):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @staticmethod
    def _argmax(values: np.ndarray[np.float64]) -> int:
        """
        :param values: List of values
        :return index of heighest value in values. Index is drawn by uniform distribution by numpy fn
        This works as np.argmax()
        """
        max_val: float = float("-inf")
        equal_values: List[int] = []

        for i in range(len(values)):
            if values[i] > max_val:
                max_val = values[i]
                equal_values = [i]
            elif values[i] == max_val:
                equal_values.append(i)
        return np.random.choice(equal_values)
