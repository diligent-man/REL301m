from .kArmAgent import kArmAgent
from ..environment import _reward, _obs

__all__ = ["GreedyKArmAgent"]


class GreedyKArmAgent(kArmAgent):
    def agent_step(self, reward: _reward, obs: _obs) -> int:
        """
        Update value of respective state and select new action based on greedy mechanism

        :param reward: the reward the agent recieved from the environment after taking the last action.
        :param obs: does not take into account in k-armed bandit problem
        :return current_action
        """
        self.arm_counter[self.last_action] += 1

        self.step_size = 1 / self.arm_counter[self.last_action]
        self.values[self.last_action] = self.values[self.last_action] + self.step_size * (reward - self.values[self.last_action])

        current_action = self._argmax(self.values)
        self.last_action = current_action
        return current_action
