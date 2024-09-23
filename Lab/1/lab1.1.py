import numpy as np
from typing import List

import numpy.random


class EpsilonGreedyAgent(object):
    def __init__(self,
                 num_actions: int = 10,
                 eps: float = .1
                 ) -> None:
        self.num_actions: int = num_actions
        self.eps: float = eps
        self.step_size: float = 0
        self.action_values: np.ndarray = np.zeros(num_actions)
        self.action_counts: np.ndarray = np.zeros(num_actions)

    def __argmax(self) -> int:
        """
        :param action_values: List of action_values
        :return index of heighest value in action_values. Index is drawn by uniform distribution by numpy fn
        This works as np.argmax()
        """
        max_val: float = float("-inf")
        equal_values: List[int] = []

        for i in range(len(self.action_values)):
            if self.action_values[i] > max_val:
                max_val = self.action_values[i]
                equal_values = [i]
            elif self.action_values[i] == max_val:
                equal_values.append(i)
        return np.random.choice(equal_values)

    def select_action(self) -> int:
        exploration_threshold = np.random.random(size=1)
        action = self.__argmax() if exploration_threshold > self.eps else \
                 np.random.choice(range(self.num_actions))
        return action

    def update_value(self, action, reward: float):
        self.action_counts[action] += 1
        self.step_size = 1 / self.action_counts[action]
        self.action_values[action] = self.action_values[action] + self.step_size * (reward - self.action_values[action])


class MultiArmedBandit(object):
    def __init__(self, num_arms: int = 10) -> None:
        self.num_arms = num_arms
        self.true_action_values = np.random.randn(num_arms)

    def get_reward(self, action: int) -> float:
        return self.true_action_values[action] + np.random.randn()


def main() -> None:
    agent: EpsilonGreedyAgent = EpsilonGreedyAgent()
    env: MultiArmedBandit = MultiArmedBandit()

    total_reward = 0.
    for step in range(5000):
        action: int = agent.select_action()
        reward: float = env.get_reward(action)
        agent.update_value(action, reward)
        total_reward += reward

    print(f"Total reward: {total_reward}")
    print(f"Estimated action values:", agent.action_values)
    return None


if __name__ == '__main__':
    main()