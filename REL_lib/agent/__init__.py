from typing import Union
from .TDAgent import TDAgent
from .kArmAgent import kArmAgent
from .GreedyAgent import GreedyKArmAgent
from .EspilonGreedyAgent import EpsilonGreedyKArmAgent
from .EpsilonGreedyAgentConstantStepsize import EpsilonGreedyKArmAgentConstantStepsize

AGENT_CLASS = TDAgent | kArmAgent | GreedyKArmAgent | EpsilonGreedyKArmAgent | EpsilonGreedyKArmAgentConstantStepsize

__all__ = [
    "TDAgent",
    "kArmAgent",
    "GreedyKArmAgent",
    "EpsilonGreedyKArmAgent",
    "EpsilonGreedyKArmAgentConstantStepsize",
    "AGENT_CLASS"
]
