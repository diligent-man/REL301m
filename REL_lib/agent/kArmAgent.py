import numpy as np

from typing import List, Dict
from .BaseAgent import BaseAgent
from ..environment import _reward, _obs

__all__ = ["kArmAgent"]


class kArmAgent(BaseAgent):
    """
    agent does *no* learning, selects action 0 always

    num_actions: number of arms in k-armed bandits problem
    initial_value: initial value of values
    values: list of numbers tracking value of selected action by each time step
    epsilon: check exploration-exploitation trade-off
    last_action: index of last action for calculating q_value
    arm_counter: counter for each arm when it is selected at current time step
    """
    num_actions: int
    initial_value: float
    values: np.ndarray[np.float64]
    step_size: float
    epsilon: float
    last_action: int
    arm_counter: np.ndarray[np.int32]

    def __init__(self,
                 num_actions: int = None,
                 initial_value: float = None,
                 values: np.ndarray[np.float64] = None,
                 step_size: float = None,
                 epsilon: float = None,
                 last_action: int = None,
                 arm_counter: np.ndarray[np.int32] = None
                 ) -> None:
        super().__init__()

        if None in (num_actions, initial_value, values, step_size, epsilon, last_action, initial_value, arm_counter):
            self.agent_init()
            print("Agent has been initialized with default paras")
        else:
            self.last_action = last_action
            self.num_actions = num_actions
            self.values = values
            self.step_size = step_size
            self.epsilon = epsilon
            self.initial_value = initial_value
            self.arm_counter = arm_counter
            print("Agent has been initialized with customized paras")

    def __repr__(self) -> str:
        return f"""num_actions: {self.num_actions}
arm_counter: {self.arm_counter}
initial_value: {self.initial_value}
values: {self.values}
step_size: {self.step_size}
epsilon: {self.epsilon}
last_action: {self.last_action}"""

    def agent_init(self, agent_info: Dict | None = None) -> None:
        """Setup for the agent called when the experiment first starts."""

        # if "actions" in agent_info:
        #     self.num_actions = agent_info["actions"]

        # if "state_array" in agent_info:
        #     self.values = agent_info["state_array"]
        if agent_info is None:
            agent_info = {}

        self.num_actions = agent_info.get("num_actions", 2)
        self.initial_value = agent_info.get("initial_value", 0.0)

        if agent_info.get("values", None) is not None:
            self.values = agent_info["values"]
            self.num_actions = len(self.values)
        else:
            self.values = np.ones(shape=self.num_actions, dtype=np.float64) * self.initial_value

        self.arm_counter: List[float] = agent_info.get("arm_counter", np.zeros(shape=self.num_actions, dtype=np.int32))
        self.step_size = agent_info.get("step_size", 0.1)
        self.epsilon = agent_info.get("epsilon", 0.0)
        self.last_action = agent_info.get("last_action", 0)
        return None

    def agent_start(self, obs: _obs) -> int:
        """The first method called when the experiment starts, called after the environment starts.
        :param obs: the observation of the current environment
        :return The first action the agent takes.
        """
        self.last_action = np.random.choice(self.num_actions)
        return self.last_action

    def agent_step(self, reward: _reward, obs: _obs) -> int:
        """A step taken by the agent.
        :param reward: the reward received for taking the last action taken
        :param obs: the state observation from the env's step based,
                    where the agent ended up after the last step
        :return The action the agent is taking.
        """
        # local_action = 0  # choose the action here
        self.last_action: int = np.random.choice(self.num_actions)
        return self.last_action

    def agent_end(self, reward: _reward): raise NotImplementedError

    def agent_cleanup(self): raise NotImplementedError

    def agent_message(self, message): raise NotImplementedError
