import numpy as np
from typing import Tuple

class GridWorld(object):
    def __init__(self):
        self.grid_size: Tuple[int, int] = (3, 3)
        self.num_actions: int = 4  # A = {left, right, up, down}
        self.rewards: np.ndarray = np.array([[0, 0, 0],
                                             [0, 0, 0],
                                             [0, 1, 0]]
                                            )

        def get_reward(self, state):
            pass


class ValueFunction(object):
    def __init__(self, grid_size: Tuple[int, int]):
        self.values = None

    def update_value(self):
        pass

    def get_value(self):
        pass


def main() -> None:
    grid_world = GridWorld()
    value_fn = ValueFunction(grid_world.grid_size)


    return None


if __name__ == '__main__':
    main()