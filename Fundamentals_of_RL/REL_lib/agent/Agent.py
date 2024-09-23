import numpy as np
from .BaseAgent import BaseAgent
from typing import List, Dict

__all__ = ["Agent"]


class Agent(BaseAgent):
    """
    agent does *no* learning, selects action 0 always

    num_actions: number of arms in k-armed bandits problem
    initial_value: initial value of q_values
    q_values: list of numbers tracking value of selected action by each time step
    epsilon: check exploration-exploitation trade-off
    last_action: index of last action for calculating q_value
    arm_count: counter for each arm when it is selected at current time step
    """
    num_actions: int
    initial_value: float
    q_values: np.ndarray[np.float64]
    step_size: float
    epsilon: float
    last_action: int
    arm_count: np.ndarray[np.int32]

    def __init__(self,
                 num_actions: int = None,
                 initial_value: float = None,
                 q_values: np.ndarray[np.float64] = None,
                 step_size: float = None,
                 epsilon: float = None,
                 last_action: int = None,
                 arm_count: np.ndarray[np.int32] = None
                 ) -> None:
        super().__init__()

        if None in (num_actions, initial_value, q_values, step_size, epsilon, last_action, initial_value, arm_count):
            self.agent_init()
            print("Agent has been initialized with default paras")
        else:
            self.last_action = last_action
            self.num_actions = num_actions
            self.q_values = q_values
            self.step_size = step_size
            self.epsilon = epsilon
            self.initial_value = initial_value
            self.arm_count = arm_count
            print("Agent has been initialized with customized paras")

    def agent_init(self, agent_info: Dict = None) -> None:
        """Setup for the agent called when the experiment first starts."""

        # if "actions" in agent_info:
        #     self.num_actions = agent_info["actions"]

        # if "state_array" in agent_info:
        #     self.q_values = agent_info["state_array"]
        if agent_info is None:
            agent_info = {}

        self.num_actions = agent_info.get("num_actions", 2)
        self.initial_value = agent_info.get("initial_value", 0.0)

        if agent_info.get("q_values", None) is not None:
            self.q_values = agent_info["q_values"]
            self.num_actions = len(self.q_values)
        else:
            self.q_values = np.ones(shape=self.num_actions, dtype=np.float64) * self.initial_value

        self.arm_count: List[float] = agent_info.get("arm_count", np.zeros(shape=self.num_actions, dtype=np.int32))
        self.step_size = agent_info.get("step_size", 0.1)
        self.epsilon = agent_info.get("epsilon", 0.0)
        self.last_action = agent_info.get("last_action", 0)
        return None

    def agent_start(self, observation: int):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's evn_start function.
                                    does not take into account in k-arm bandits

        Returns:
            The first action the agent takes.

        """
        self.last_action = np.random.choice(self.num_actions)  # set first action to 0
        return self.last_action

    def agent_step(self, reward: float, observation: np.ndarray) -> int:
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        # local_action = 0  # choose the action here
        self.last_action = np.random.choice(self.num_actions)
        return self.last_action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        pass

    def agent_cleanup(self):
        """Cleanup done after the agent ends."""
        pass

    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """
        pass

    def __repr__(self) -> str:
        return f"""num_actions: {self.num_actions}
arm_count: {self.arm_count}
initial_value: {self.initial_value}
q_values: {self.q_values}
step_size: {self.step_size}
epsilon: {self.epsilon}
last_action: {self.last_action}
"""
