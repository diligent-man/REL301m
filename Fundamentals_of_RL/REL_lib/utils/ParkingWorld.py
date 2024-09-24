import numpy as np
from typing import List, Tuple

__all__ = ["ParkingWorld"]


class ParkingWorld:
    def __init__(self,
                 num_spaces=10,
                 num_prices=4,
                 price_factor=0.1,
                 occupants_factor=1.0,
                 null_factor=1/3):
        self.__num_spaces = num_spaces
        self.__num_prices = num_prices
        self.__occupants_factor = occupants_factor
        self.__price_factor = price_factor
        self.__null_factor = null_factor
        self.__S = [num_occupied for num_occupied in range(num_spaces + 1)]
        self.__A = list(range(num_prices))

    @property
    def A(self):
        return list(self.__A)

    @property
    def num_spaces(self):
        return self.__num_spaces

    @property
    def num_prices(self):
        return self.num_prices

    @property
    def S(self):
        return list(self.__S)

    def __repr__(self) -> str:
        return f"""Num spaces: {self.num_spaces}
    Num prices: {self.num_prices}
    Occupants factor: {self.__occupants_factor}
    Price factor: {self.__price_factor}
    Null factor: {self.__null_factor}
    State space: {self.__S}
    Action space: {self.__A}
    """

    @staticmethod
    def prettify_transitions(state: int, action: int, transitions: np.ndarray) -> None:
        for s_prime, (reward, prob) in enumerate(transitions):
            print(f"p(S\'={s_prime}, R={reward:<5.3} | S={state}, A={action}) = {prob:.2}")

    def transitions(self, s: int, a: int) -> np.ndarray:
        """
        :param s: current state
        :param a: taken action
        :return: 2D array.
            1st col is reward for transitioning from current to next state i,
            2nd col is conditional prob for that transition, which is p(s', r| s, a)
        """
        return np.array([[r, self.p(s_, r, s, a)] for s_, r in self.support(s)])

    def support(self, s: int) -> List[Tuple[int, float]]:
        """
        :param s: current state
        :return: List of tuple of (next_state, reward_from_current_to_next_state)
        """
        return [(s_, self.reward(s, s_)) for s_ in self.__S]

    def p(self, s_, r, s, a):
        if r != self.reward(s, s_):
            return 0
        else:
            center = (1 - self.__price_factor
                      ) * s + self.__price_factor * self.__num_spaces * (
                          1 - a / self.__num_prices)
            emphasis = np.exp(
                -abs(np.arange(2 * self.__num_spaces) - center) / 5)
            if s_ == self.__num_spaces:
                return sum(emphasis[s_:]) / sum(emphasis)
            return emphasis[s_] / sum(emphasis)

    def reward(self, s: int, s_: int) -> float:
        """
        :param s: current state 
        :param s_: next state
        :return: reward of transitioning from current stet to next state
        """
        return self.state_reward(s) + self.state_reward(s_)

    def state_reward(self, s: int) -> float:
        if s == self.__num_spaces:
            return self.__null_factor * s * self.__occupants_factor
        else:
            return s * self.__occupants_factor

    def random_state(self):
        return np.random.randint(self.__num_prices)

    def step(self, s, a):
        probabilities = [
            self.p(s_, self.reward(s, s_), s, a) for s_ in self.__S
        ]
        return np.random.choice(self.__S, p=probabilities)

