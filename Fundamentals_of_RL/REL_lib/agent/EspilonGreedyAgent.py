import numpy as np
from .Agent import Agent

__all__ = ["EpsilonGreedyAgent"]


class EpsilonGreedyAgent(Agent):
    def agent_step(self, reward: float, observation: np.ndarray) -> int:
        """
        Arguments:
            reward (float): the reward the agent recieved from the environment after taking the last action.
            observation: does not take into account in k-armed bandit problem

        Returns:
            current_action -- int, the action chosen by the agent at the current time step.

        Useful Class Variables
            q_values : An array with what the agent believes each of the values of the arm are.
            arm_count : An array with a count of the number of times each arm has been pulled.
            last_action : The action that the agent took on the previous time step
            epsilon : The probability an epsilon greedy agent will explore (ranges between 0 and 1)

        Update steps
            1/ Increment arm_count of last action
            2/ Update step_size by 1 / arm_count of last_action
            3/ Update q_values of last_action by update rule
            4/ Compare with exploration_threshold
                if self.exploration_threshold > self.epsilon:
                    Take argmax of q_values
                else:
                    Pick random action

        Update q_values rule:
            new_q_values = old_q_values + step_size * [current_reward - old_q_values]
            step_size = 1 / total_step_of_selected_action
        """
        exploration_threshold = np.random.random(size=1)

        self.arm_count[self.last_action] += 1
        self.step_size = 1 / self.arm_count[self.last_action]

        self.q_values[self.last_action] = self.q_values[self.last_action] + self.step_size * (reward - self.q_values[self.last_action])
        current_action = self._argmax(self.q_values) if exploration_threshold > self.epsilon else np.random.choice(range(self.num_actions))

        self.last_action = current_action
        return current_action
