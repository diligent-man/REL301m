import os
import shutil
import numpy as np

from typing import Tuple
from semester_8.REL301m.REL_lib import (
    TDAgent, AGENT_CLASS,
    CliffWalkEnv, ENV_CLASS,
    RLGlue, GridWorldManager,
    evaluate_policy_with_TD
)


class GlobalVar:
    def __init__(self):
        self.theta: float = 1e-2

        self.grid_h: int = 4
        self.grid_w: int = 12

        self.num_states = self.grid_w * self.grid_h
        self.num_actions = 4

        self.V_to_optimze: np.ndarray = np.zeros((self.num_states, ))
        self.V_of_optimal_pi: np.ndarray = np.load("val_fn_with_optimal_pi.npy")
        self.V_of_safe_pi: np.ndarray = np.load("val_with_safe_pi.npy")

        self.uniform_pi: np.ndarray = np.ones((self.num_states, self.num_actions)) / self.num_actions
        self.stride_along_cliff_pi = self.__contrive_optimal_pi("stride_along_cliff")
        self.modified_pi = self.__contrive_optimal_pi("modified")
        self.stochastic_pi = self.__contrive_optimal_pi("stochastic")

        save_path_root: str = os.path.join(os.getcwd(), "TD_training_result")
        os.makedirs(save_path_root, exist_ok=True)

        self.experiment_path: str = os.path.join(save_path_root, "Tabular_TD0")
        if os.path.exists(self.experiment_path):
            shutil.rmtree(self.experiment_path)
        os.makedirs(self.experiment_path, exist_ok=False)

    def __contrive_optimal_pi(self, cond: str) -> np.ndarray:
        """
        :param cond: ("stride_along_cliff", "modified", "stochastic")
        :return: contrived policy
        """
        assert cond in ("stride_along_cliff", "modified", "stochastic"), "Provided cond is unavailable"
        contrived_policy = self.uniform_pi.copy()

        if cond == "stride_along_cliff":
            penultimate_row_idx: Tuple[int, int] = ((self.grid_h-2) * self.grid_w, (self.grid_h-1) * self.grid_w)

            contrived_policy[penultimate_row_idx[-1]] = [0, 0, 1, 0]  # next to last goal_loc must go down
            contrived_policy[penultimate_row_idx[-1]+1] = [1, 0, 0, 0]  # start_loc must go up

            for i in range(*penultimate_row_idx):
                contrived_policy[i] = [0, 0, 0, 1]
        elif cond == "modified":
            for i in range(self.grid_h-1, 0, -1):
                contrived_policy[i * self.grid_w] = [1, 0, 0, 0]

            for j in range(self.grid_w):
                contrived_policy[j] = [0, 0, 0, 1]

            for i in range(self.grid_h-1):
                contrived_policy[i * self.grid_w + (self.grid_w-1)] = [0, 0, 1, 0]
        else:
            penultimate_row_idx: Tuple[int, int] = ((self.grid_h - 2) * self.grid_w, (self.grid_h - 1) * self.grid_w)

            contrived_policy[penultimate_row_idx[-1]] = [0.1/3., 0.1/3., 0.9, 0.1/3.]
            contrived_policy[penultimate_row_idx[-1] + 1] = [0.9, 0.1/3., 0.1/3., 0.1/3.]

            for i in range(*penultimate_row_idx):
                contrived_policy[i] = [0.1 / 3., 0.1 / 3., 0.1 / 3., 0.9]
        return contrived_policy


def main() -> None:
    globalVar = GlobalVar()

    agent: AGENT_CLASS = TDAgent()
    agent_info = {"policy": globalVar.uniform_pi, "values": globalVar.V_to_optimze}

    env: ENV_CLASS = CliffWalkEnv()
    env_info = {"grid_height": globalVar.grid_h, "grid_width": globalVar.grid_w}

    rLGlue = RLGlue(agent, env)
    rLGlue.rl_init(agent_info, env_info)

    gridWorldManager = GridWorldManager(agent_info, env_info,
                                        experiment_name="demo",
                                        true_values_file="optimal_policy_value_fn.npy"
                                        )

    V = evaluate_policy_with_TD(rLGlue, gridWorldManager, globalVar.theta, 1, 10, globalVar.experiment_path)
    # TODO: Check train process
    return None


if __name__ == '__main__':
    main()