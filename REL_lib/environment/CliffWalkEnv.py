from typing import Dict, Tuple, List
from .BaseEnv import (
    _reward, _obs, _term,
    BaseEnvironment
)

__all__ = ['CliffWalkEnv']


class CliffWalkEnv(BaseEnvironment):
    __reward: _reward = None
    __obs: _obs = None
    __term: _term = None

    __grid_h: int = None
    __grid_w: int = None
    __start_loc: Tuple[int, int] = None
    __goal_loc: Tuple[int, int] = None
    __cliff: List[Tuple[int, int]] = None

    __agent_loc: Tuple[int, int] = None
    """
    Frame of reference: (0, 0) coord is top-left cell
    State: (reward, state, termination)
    Action: (up, left, down, right) == (0, 1, 2, 3)
    
    """
    def __init__(self):
        super().__init__()

    @property
    def agent_loc(self) -> Tuple[int, int]:
        return self.__agent_loc

    @property
    def cliff(self) -> List[Tuple[int, int]]:
        return self.__cliff

    def __repr__(self) -> str:
        return f"""Grid height: {self.__grid_h}
Grid width: {self.__grid_w}
Start location: {self.__start_loc}
Goal location: {self.__goal_loc}
Cliff: {self.__cliff}"""

    def __isInBounds(self, x: int, y: int) -> bool:
        if x in range(0, self.__grid_h) and y in range(0, self.__grid_w):
            return True
        else:
            return False

    def __coord_to_spatial(self, coord: Tuple[int, int]) -> int:
        """
        :param coord: (x, y) with row-major format
        :return: flattened single index of current location
        """
        return coord[0] * self.__grid_w + coord[1]

    def env_init(self, env_info: Dict = None) -> None:
        """Setup for the environment called when the experiment first starts.
        Note:
            Initialize a tuple with the reward, first state, boolean
            indicating if it's terminal.
        """
        if env_info is None:
            env_info: Dict = {}

        self.__grid_h = env_info.get("grid_height", 4)
        self.__grid_w = env_info.get("grid_width", 12)

        # Define frame of reference with row-major Numpy convention.
        self.__start_loc = (self.__grid_h-1, 0)
        self.__goal_loc = (self.__grid_h-1, self.__grid_w-1)

        # Cliff is the same row of starting and goal loc
        self.__cliff = [(self.__grid_h - 1, i) for i in range(1, (self.__grid_w-1))]
        return None

    def env_start(self) -> int:
        """
        The first method called when the episode starts,
        called before the agent starts.

        :return: first state of the environment
        """
        self.__agent_loc = self.__start_loc
        self._reward_obs_term = (0, self.__coord_to_spatial(self.__agent_loc), False)
        return self._reward_obs_term[1]

    def env_step(self, action: int) -> List[int | bool]:
        """
        :param action: selected action
        :return: (next state, reward, termination): (float, state, Boolean)
        """
        x, y = self.__agent_loc

        if action == 0:
            x -= 1  # Up
        elif action == 1:
            y -= 1  # Left
        elif action == 2:
            x += 1  # Down
        elif action == 3:
            y += 1  # Right
        else:
            prompt = f"{action} not in recognized actions [0: Up, 1: Left, 2: Down, 3: Right]!"
            raise Exception(prompt)

        self.__agent_loc = (x, y) if self.__isInBounds(x, y) else self.__agent_loc

        # by default, assume -1 reward per step and that we did not terminate
        reward_obs_term = [-1, self.__coord_to_spatial(self.__agent_loc), False]

        if self.__agent_loc in self.__cliff:
            self.__agent_loc = self.__start_loc
            reward_obs_term[:2] = (-100, self.__coord_to_spatial(self.__agent_loc))
        elif (x, y) == self.__goal_loc:
            reward_obs_term[-1] = True

        self._reward_obs_term = reward_obs_term
        return reward_obs_term

    def env_message(self, message: str):
        raise NotImplementedError

    def env_cleanup(self):
        """Reset agent"""
        self.__agent_loc = self.__start_loc
