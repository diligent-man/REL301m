from .TenArmEnv import TenArmEnv
from .ParkingWorld import ParkingWorld

ENV_CLASS = TenArmEnv | ParkingWorld

__all__ = ["TenArmEnv", "ParkingWorld", "ENV_CLASS"]
