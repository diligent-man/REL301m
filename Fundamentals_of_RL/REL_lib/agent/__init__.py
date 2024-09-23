from .Agent import Agent
from .GreedyAgent import GreedyAgent
from .EspilonGreedyAgent import EpsilonGreedyAgent
from .EpsilonGreedyAgentConstantStepsize import EpsilonGreedyAgentConstantStepsize

AGENT_CLASS = Agent | GreedyAgent | EpsilonGreedyAgent | EpsilonGreedyAgentConstantStepsize

__all__ = [
    "Agent",
    "GreedyAgent",
    "EpsilonGreedyAgent",
    "EpsilonGreedyAgentConstantStepsize",
    "AGENT_CLASS"
]
