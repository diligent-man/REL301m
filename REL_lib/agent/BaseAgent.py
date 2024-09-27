import numpy as np
from typing import Dict, List
from abc import ABCMeta, abstractmethod


__all__ = ["BaseAgent"]


class BaseAgent(metaclass=ABCMeta):
    """Implements the agent for an RL-Glue environment."""

    def __init__(self):
        pass

    @abstractmethod
    def agent_init(self, agent_info: Dict):
        """Setup for the agent called when the experiment first starts."""
        pass

    @abstractmethod
    def agent_start(self, observation: np.ndarray):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        pass

    @abstractmethod
    def agent_step(self, reward: float, observation: np.ndarray):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        pass

    @abstractmethod
    def agent_end(self, reward: float):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the terminal state.
        """
        pass

    @abstractmethod
    def agent_cleanup(self):
        """Cleanup done after the agent ends."""
        pass

    @abstractmethod
    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @staticmethod
    def _argmax(q_values: np.ndarray[np.float64]) -> int:
        """
        :param q_values: List of q_values
        :return index of heighest value in q_values. Index is drawn by uniform distribution by numpy fn
        This works as np.argmax()
        """
        max_val: float = float("-inf")
        equal_values: List[int] = []

        for i in range(len(q_values)):
            if q_values[i] > max_val:
                max_val = q_values[i]
                equal_values = [i]
            elif q_values[i] == max_val:
                equal_values.append(i)
        return np.random.choice(equal_values)
