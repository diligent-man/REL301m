import os
import numpy as np

from typing import Dict, Tuple
from matplotlib import pyplot as plt

plt.switch_backend("tkagg")

__all__ = ["GridWorldManager"]


class GridWorldManager(object):
    policy: np.ndarray
    grid_h: int
    grid_w: int

    cmap = plt.cm.viridis
    cmap.set_bad("black", 1.)

    values_table: np.ndarray = None
    optimal_values_file: str = None
    experiment_name: str = None

    def __init__(self,
                 agent_info: Dict,
                 env_info: Dict,
                 experiment_name: str,
                 optimal_values_file: str = None,
                 ) -> None:
        assert "grid_height" in env_info.keys(), "Require grid height in env_info"
        assert "grid_width" in env_info.keys(), "Require grid width in env_info"
        assert "policy" in agent_info.keys(), "Require policy in agent_info"

        self.policy = agent_info["policy"]
        self.grid_h, self.grid_w = env_info["grid_height"], env_info["grid_width"]
        self.optimal_values_file = optimal_values_file
        self.experiment_name = experiment_name

    def compute_values_table(self, values: np.ndarray) -> None:
        self.values_table = np.empty((self.grid_h, self.grid_w))
        self.values_table.fill(np.nan)

        for state in range(len(values)):
            coord: Tuple[int, int] = np.unravel_index(state, (self.grid_h, self.grid_w))
            self.values_table[coord] = values[state]

    def compute_RMSVE(self):
        return (np.sqrt(np.nanmean((self.values_table - self.optimal_values) ** 2)))

    def plot_training_result(self,
                             values: np.ndarray,
                             episode_num: int,
                             save_path_root: str,
                             title: str = None,
                             ) -> None:
        if not hasattr(self, "fig"):
            self.fig = plt.figure(figsize=(10, 20))

            if self.optimal_values_file is not None:
                self.cmap_VE = plt.cm.Reds
                self.cmap_VE.set_bad('black', 1.)
                self.ax = self.fig.add_subplot(311)
                self.RMSVE_LOG = []
                self.optimal_values = np.load(self.optimal_values_file)
            else:
                self.optimal_values = None

        self.fig.clear()
        if self.optimal_values is not None:
            plt.subplot(311)

        self.compute_values_table(values)
        plt.xticks([])
        plt.yticks([])
        im = plt.imshow(self.values_table, cmap=self.cmap, interpolation='nearest', origin='upper')

        for state in range(self.policy.shape[0]):
            for action in range(self.policy.shape[1]):
                y, x = np.unravel_index(state, (self.grid_h, self.grid_w))
                pi = self.policy[state][action]
                if pi == 0:
                    continue
                if action == 0:
                    plt.arrow(x, y, 0, -0.5 * pi, fill=False, length_includes_head=True, head_width=0.1,
                              alpha=0.5)
                if action == 1:
                    plt.arrow(x, y, -0.5 * pi, 0, fill=False, length_includes_head=True, head_width=0.1,
                              alpha=0.5)
                if action == 2:
                    plt.arrow(x, y, 0, 0.5 * pi, fill=False, length_includes_head=True, head_width=0.1,
                              alpha=0.5)
                if action == 3:
                    plt.arrow(x, y, 0.5 * pi, 0, fill=False, length_includes_head=True, head_width=0.1,
                              alpha=0.5)
        plt.title((title + "\n") + "Predicted Values, Episode: %d" % episode_num)
        plt.colorbar(im, orientation='horizontal')

        if self.optimal_values is not None:
            plt.subplot(312)
            plt.xticks([])
            plt.yticks([])
            im = plt.imshow((self.values_table - self.optimal_values) ** 2, origin='upper', cmap=self.cmap_VE)
            plt.title('Squared Value Error: $(v_{\pi}(S) - \hat{v}(S))^2$')
            plt.colorbar(im, orientation='horizontal')
            self.RMSVE_LOG.append((episode_num, self.compute_RMSVE()))

            plt.subplot(313)
            plt.plot([x[0] for x in self.RMSVE_LOG], [x[1] for x in self.RMSVE_LOG])
            plt.xlabel("Episode")
            plt.ylabel("RMSVE", rotation=0, labelpad=20)
            plt.title("Root Mean Squared Value Error")
        self.fig.canvas.draw()

        save_path = os.path.join(save_path_root, f"{self.experiment_name}_{episode_num}.png")
        self.fig.savefig(save_path)

    def run_tests(self, values, RMSVE_threshold):
        assert self.optimal_values is not None, "This function can only be called once the true values are given during " + \
                                             "runtime."
        self.compute_values_table(values)
        mask = ~(np.isnan(self.values_table) | np.isnan(self.optimal_values))
        if self.compute_RMSVE() < RMSVE_threshold and np.allclose(self.optimal_values[mask], self.values_table[mask]):
            pass
        else:
            assert False

    def __del__(self):
        print("Destructor called")

