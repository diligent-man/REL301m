from types import UnionType
from typing import Tuple, Dict
from abc import ABCMeta, abstractmethod

__all__ = ["BaseEnvironment", "_reward", "_obs", "_term"]

import numpy as np

_reward =  int | float | None
_obs = int | Tuple | np.ndarray | None
_term = bool | None


class BaseEnvironment(metaclass=ABCMeta):
    _reward_obs_term: Tuple[_reward, _obs, _term]

    def __init__(self) -> None:
        self._reward_obs_term = (None, None, None)

    @abstractmethod
    def env_init(self, env_info: Dict):
        """
        Setup for the environment called when the experiment first starts.
        """
        raise NotImplementedError

    @abstractmethod
    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts. Returns 1st state obs from env
        """
        raise NotImplementedError

    @abstractmethod
    def env_step(self, action: int):
        """
        A step taken by the environment. Returns reward_obs_term tuple
        """
        raise NotImplementedError

    @abstractmethod
    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        raise NotImplementedError

    @abstractmethod
    def env_message(self, message: str):
        """
        Query some info of env
        """
        raise NotImplementedError
