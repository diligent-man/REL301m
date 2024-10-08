import datetime
import overrides
from typing import Dict, Optional, List, Tuple
from .BaseEnv import BaseEnvironment
from gymnasium.envs.toy_text import FrozenLakeEnv
from gymnasium.envs.toy_text.utils import categorical_sample


__all__ = ["CustomFrozenLakeEnv"]


class CustomFrozenLakeENV(FrozenLakeEnv, BaseEnvironment):
    def __init__(self,
                 render_mode: Optional[str] = None,
                 desc: List[str] = None,
                 map_name: str = "4x4",
                 is_slippery: bool = True,
                 seed: int = None
                 ):
        self.__seed = seed
        super().__init__(render_mode, desc, map_name, is_slippery)

    @overrides.override
    def step(self, action: int) -> Tuple[float, int, bool, float]:
        """
        Perform a step in the environment.
        (Remove False in return Tuple cuz of not knowing what )
        """
        if not hasattr(self, "s"):
            print("env_start() should be called at first")

        transitions = self.P[self.s][action]
        i = categorical_sample([t[0] for t in transitions], self.np_random)

        prob, state, reward, term = transitions[i]
        self.s = state
        self.lastaction = action

        if self.render_mode == "human":
            self.render()
        self._reward_obs_term = (reward, int(state), term)
        return (reward, int(state), term, prob)

    @overrides.override
    def env_init(self, env_info: Dict = None) -> None:
        """
        :param env_info: info to change any attr of left-most inherited class
        """
        if env_info is None:
            env_info = {
                "render_mode": None,
                "desc": None,
                "map_name": "4x4",
                "is_slippery": True
            }
        super().__init__(**env_info)
        return None

    @overrides.override
    def env_start(self) -> int:
        """first state of the environment"""
        self.reset(seed=int(datetime.datetime.now().timestamp()))
        self._reward_obs_term = (0, self.s, False)
        return self._reward_obs_term[1]

    @overrides.override
    def env_step(self, action: int) -> Tuple[float, int, bool, float]:
        reward, state, term, prob = self.step(action)
        self._reward_obs_term = (reward, int(state), term)
        return (reward, int(state), term, prob)

    @overrides.override
    def env_cleanup(self) -> None:
        if self.__seed is None:
            self.reset(seed=int(datetime.datetime.now().timestamp()))
        else:
            self.reset(seed=self.__seed)

    @overrides.override
    def env_message(self, message: str):
        raise NotImplementedError
