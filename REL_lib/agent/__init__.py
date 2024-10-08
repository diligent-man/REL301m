from .TDAgent import TDAgent
from .kArmAgent import kArmAgent
from .GreedyAgent import GreedyKArmAgent
from .EspilonGreedyAgent import EpsilonGreedyKArmAgent
from .EpsilonGreedyAgentConstantStepsize import EpsilonGreedyKArmAgentConstantStepsize

from .FrozenLakeBellmanAgent import FrozenLakeBellmanAgent
from .FrozenLakeESControlMCAgent import FrozenLakeESControlMCAgent
from .FrozenLakeOnPolicyMCAgent import FrozenLakeOnPolicyMCAgent
from .FrozenLakeSARSAAgent import FrozenLakeSARSAAgent

AGENT_CLASS = TDAgent | kArmAgent | GreedyKArmAgent | EpsilonGreedyKArmAgent | EpsilonGreedyKArmAgentConstantStepsize |\
              FrozenLakeBellmanAgent | FrozenLakeESControlMCAgent | FrozenLakeOnPolicyMCAgent | FrozenLakeSARSAAgent


__all__ = [
    "TDAgent",
    "kArmAgent",
    "GreedyKArmAgent",
    "EpsilonGreedyKArmAgent",
    "EpsilonGreedyKArmAgentConstantStepsize",

    "FrozenLakeBellmanAgent",
    "FrozenLakeESControlMCAgent",
    "FrozenLakeOnPolicyMCAgent",
    "FrozenLakeSARSAAgent",

    "AGENT_CLASS"
]
