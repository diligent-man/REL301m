"""
This env is used for training 10-arm bandit problem
"""
import numpy as np

from typing import Tuple
from .BaseEnv import BaseEnvironment

__all__ = ['TenArmEnv']


class TenArmEnv(BaseEnvironment):
    """Implements the environment for an 10-armed bandits

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    Args:
        arms: reward for 1o arms being sampled from normal distribution
        seed: randomness setting
    """
    arms: np.ndarray[np.float64] = None
    seed: int = None

    def __init__(self,
                 reward: int | float = None,
                 obs: int | Tuple = None,
                 term: bool = None
                 ) -> None:
        super().__init__(reward, obs, term)

    def env_init(self, env_info=None) -> None:
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """
        if env_info is None:
            env_info = {}

        self.seed: int = env_info.get("random_seed", None)
        np.random.seed(self.seed)

        self.arms: np.ndarray[np.float64] = np.random.randn(10)  # sampled from normal dist
        self._reward_obs_term: Tuple = (0.0, 0, False)

    def env_start(self) -> int:
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        return self._reward_obs_term[1]

    def env_step(self, action: int) -> Tuple[float, int, bool]:
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        # if action == 0:
        #     if np.random.random() < 0.2:
        #         reward = 14
        #     else:
        #         reward = 6

        # if action == 1:
        #     reward = np.random.choice(range(10,14))

        # if action == 2:
        #     if np.random.random() < 0.8:
        #         reward = 174
        #     else:
        #         reward = 7
        noise = np.random.randn()

        self._reward_obs_term = (self.arms[action] + noise,  # reward for current timestep
                                self._reward_obs_term[1],  # current env state
                                False)
        return self._reward_obs_term

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        pass

    def env_message(self, message) -> str:
        """A message asking the environment for information

        Args:
            message (string): the message passed to the environment

        Returns:
            string: the response (or answer) to the message
        """
        if message == "what is the current reward?":
            return "{}".format(self._reward_obs_term[0])

        # else
        return "I don't know how to respond to your message"

    def __repr__(self) -> str:
        return f"""Reward, obs, term: {self._reward_obs_term},
Arms: {self.arms},
Seed: {self.seed}
"""
