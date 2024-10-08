import numpy as np

from typing import Dict, List
from abc import ABCMeta, abstractmethod


__all__ = ["GPIAgent"]


class GPIAgent(metaclass=ABCMeta):
    """
    Base class for GPI-based agents.
    """
    def __init__(self):
        pass

    def _eval_policy(self, policy: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _improve_policy(self, policy: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def train(self, num_iters: int) -> np.ndarray:
        """
        :param num_iters: number of iterations to train agent
        :return trained policy
        """
        raise NotImplementedError
