"""
Glues together an experiment, agent, and environment.
"""
from typing import Dict, Tuple
from .agent import AGENT_CLASS
from .environment import ENV_CLASS

__all__ = ["RLGlue"]


class RLGlue:
    """
    args:
        env_name (obj): the name of the module where the Environment class can be found
        agent_name (obj): the name of the module where the Agent class can be found
        total_reward

    """
    agent: AGENT_CLASS
    environment: ENV_CLASS
    total_reward = 0.
    num_steps = 0
    last_action = None
    num_episodes = 0

    def __init__(self,
                 agent_class: AGENT_CLASS,
                 env_class: ENV_CLASS
                 ) -> None:
        assert isinstance(agent_class, AGENT_CLASS), "Provided agent class is unavailable"
        assert isinstance(env_class, ENV_CLASS), "Provided env class is unavailable"

        self.agent: AGENT_CLASS = agent_class
        self.environment: ENV_CLASS = env_class

    def rl_init(self,
                agent_info: Dict = None,
                env_info: Dict = None
                ) -> None:
        """Initial method called when RLGlue experiment is created"""
        self.agent.agent_init() if agent_info is None else self.agent.agent_init(agent_info)
        self.environment.env_init() if env_info is None else self.environment.env_init(env_info)

        self.total_reward = 0.0
        self.num_steps = 0
        self.num_episodes = 0

    def rl_start(self) -> Tuple[int, int]:
        """Starts RLGlue experiment"""
        last_state = self.environment.env_start()
        self.last_action = self.agent.agent_start(last_state)

        observation = (last_state, self.last_action)
        return observation

    def rl_agent_start(self, observation):
        """Starts the agent.

        Args:
            observation: The first observation from the environment

        Returns:
            The action taken by the agent.
        """
        return self.agent.agent_start(observation)

    def rl_agent_step(self, reward, observation):
        """Step taken by the agent

        Args:
            reward (float): the last reward the agent received for taking the
                last action.
            observation : the state observation the agent receives from the
                environment.

        Returns:
            The action taken by the agent.
        """
        return self.agent.agent_step(reward, observation)

    def rl_agent_end(self, reward):
        """Run when the agent terminates

        Args:
            reward (float): the reward the agent received when terminating
        """
        self.agent.agent_end(reward)

    def rl_env_start(self):
        """Starts RL-Glue environment.

        Returns:
            (float, state, Boolean): reward, state observation, boolean
                indicating termination
        """
        self.total_reward = 0.0
        self.num_steps = 1

        this_observation = self.environment.env_start()

        return this_observation

    def rl_env_step(self, action):
        """Step taken by the environment based on action from agent

        Args:
            action: Action taken by agent.

        Returns:
            (float, state, Boolean): reward, state observation, boolean
                indicating termination.
        """
        ro = self.environment.env_step(action)
        (this_reward, _, terminal) = ro

        self.total_reward += this_reward

        if terminal:
            self.num_episodes += 1
        else:
            self.num_steps += 1

        return ro

    def rl_step(self) -> Tuple:
        """
        Step taken by RLGlue, takes env step and either step or end by agent.

        :return reward_obs_act_term (float, state, action, Boolean)
        """
        (reward, last_state, term) = self.environment.env_step(self.last_action)
        self.total_reward += reward

        if term:
            self.num_episodes += 1
            self.agent.agent_end(reward)
            reward_obs_act_term = (reward, last_state, None, term)
        else:
            self.num_steps += 1
            self.last_action = self.agent.agent_step(reward, last_state)
            reward_obs_act_term = (reward, last_state, self.last_action, term)
        return reward_obs_act_term

    def rl_cleanup(self):
        """Cleanup done at end of experiment."""
        self.environment.env_cleanup()
        self.agent.agent_cleanup()

    def rl_agent_message(self, message):
        """Message passed to communicate with agent during experiment

        Args:
            message: the message (or question) to send to the agent

        Returns:
            The message back (or answer) from the agent

        """

        return self.agent.agent_message(message)

    def rl_env_message(self, message):
        """Message passed to communicate with environment during experiment

        Args:
            message: the message (or question) to send to the environment

        Returns:
            The message back (or answer) from the environment

        """
        return self.environment.env_message(message)

    def rl_episode(self, max_steps_this_episode: int = 0) -> bool:
        """
        :param max_steps_this_episode:
            0: step until being terminated
            !0: otherwise
        :return: if the episode should terminate
        """
        is_terminal = False
        _ = self.rl_start()

        while (not is_terminal) and \
              ((max_steps_this_episode == 0) or (self.num_steps < max_steps_this_episode)):
            rl_step_result = self.rl_step()
            is_terminal = rl_step_result[3]
        return is_terminal

    def rl_return(self):
        """The total reward

        Returns:
            float: the total reward
        """
        return self.total_reward

    def rl_num_steps(self):
        """The total number of steps taken

        Returns:
            Int: the total number of steps taken
        """
        return self.num_steps

    def rl_num_episodes(self):
        """The number of episodes

        Returns
            Int: the total number of episodes

        """
        return self.num_episodes

    def __repr__(self) -> str:
        return f"""Agent: {self.agent.__class__.__name__}
Env: {self.environment.__class__.__name__}
Total reward = {self.total_reward}
Num steps = {self.num_steps}
Last action = {self.last_action}
Num episodes = {self.num_steps}"""
