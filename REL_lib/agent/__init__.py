from .TDAgent import TDAgent
from .kArmAgent import kArmAgent
from .GreedyAgent import GreedyKArmAgent
from .EspilonGreedyAgent import EpsilonGreedyKArmAgent
from .EpsilonGreedyAgentConstantStepsize import EpsilonGreedyKArmAgentConstantStepsize

from .FrozenBellmanAgent import FrozenBellmanAgent
from .FrozenESControlMCAgent import FrozenESControlMCAgent
from .FrozenOnPolicyMCAgent import FrozenOnPolicyMCAgent


AGENT_CLASS = TDAgent | kArmAgent | GreedyKArmAgent | EpsilonGreedyKArmAgent | EpsilonGreedyKArmAgentConstantStepsize |\
              FrozenBellmanAgent | FrozenESControlMCAgent | FrozenOnPolicyMCAgent


__all__ = [
    "TDAgent",
    "kArmAgent",
    "GreedyKArmAgent",
    "EpsilonGreedyKArmAgent",
    "EpsilonGreedyKArmAgentConstantStepsize",

    "FrozenBellmanAgent",
    "FrozenESControlMCAgent",
    "FrozenOnPolicyMCAgent",

    "AGENT_CLASS"
]
