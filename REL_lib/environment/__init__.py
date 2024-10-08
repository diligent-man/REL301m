from .BaseEnv import _reward, _obs, _term
from .kArmEnv import TenArmEnv
from .ParkingWorld import ParkingWorld
from .CliffWalkEnv import CliffWalkEnv
from .CustomFrozenLakeEnv import CustomFrozenLakeENV

ENV_CLASS = TenArmEnv | ParkingWorld | CliffWalkEnv | CustomFrozenLakeENV

__all__ = [
    "_reward", "_obs", "_term",
    "TenArmEnv", "ParkingWorld", "CliffWalkEnv", "CustomFrozenLakeENV", "ENV_CLASS"
]
