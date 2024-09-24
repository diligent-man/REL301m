import numpy as np

from tqdm import tqdm
from typing import Dict, Tuple, List, Union

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from semester_8.REL301m.Fundamentals_of_RL.REL_lib import (
    AGENT_CLASS, ENV_CLASS,
    RLGlue
)

plt.switch_backend("tkagg")

__all__ = [
    "visualize_training_result",
    "visualize_best_action_chosen",
    "visualize_step_size_effect_to_q_value",
    "visualize_value_fn",
    "train"
]


def visualize_training_result(num_runs: int, num_steps: int,
                              rewards: np.ndarray,
                              average_best: Union[float, List[float]],
                              legends: List[str] = None,
                              title: str = None,
                              save_name: str = None
                              ) -> None:
    """
    :param num_runs: number of experiment an agent is going to perform
    :param num_steps: number of step for each conducted experiment
    :param rewards: can be in shape of either (num_steps, ) or (agent, num_steps)
    :param average_best: can be
        single number: user-defined threshold for average reward
        list of numbers: threshold for multiple agents
    :param legends: list of legends include names for average_best, rewards respectively
    :param title: title for plotting
    :param save_name: file name for saving plotted graph
    """
    if rewards.ndim == 1:
        assert isinstance(average_best, float), "Single value shoud be provided for single agent case"
    elif rewards.ndim == 2:
        assert isinstance(average_best, float) or rewards.shape[0] == len(average_best), \
            "You can provide either single number or list of number having len equivalent to num of agents"

        if legends is not None:
            assert 2 * rewards.shape[0] == len(legends), "Num of legends should be equal to num of agents"

    if isinstance(average_best, float):
        average_best = [average_best]

    plt.figure(figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')

    # Plot threshold
    for val in average_best:
        plt.plot([val / num_runs for _ in range(num_steps)], linestyle="--")

    # Plot reward
    for reward in rewards:
        plt.plot(reward)

    if legends is not None:
        plt.legend(legends)

    plt.title("Avg Reward of Agent") if title is None else plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Avg reward")
    plt.savefig(save_name) if save_name is not None else plt.show()
    return None


def visualize_best_action_chosen(best_action_lst: np.ndarray[np.float32],
                                 legends: List[str],
                                 title: str = None,
                                 save_name: str = None
                                 ) -> None:
    plt.figure(figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
    for lst in best_action_lst:
        plt.plot(lst)

    plt.legend(legends)
    plt.xlabel("Steps")
    plt.ylabel("% Best Arm Pulled")

    plt.title("% Best Arm Pulled") if title is None else plt.title(title)
    plt.savefig(save_name) if save_name is not None else plt.show()
    return None


def visualize_step_size_effect_to_q_value(max_expected_value: np.float64,
                                          estimated_value: np.ndarray,
                                          legends: List[str],
                                          title: str,
                                          save_name: str
                                          ) -> None:
    plt.figure(figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')

    plt.plot(estimated_value)
    plt.plot([max_expected_value for _ in range(len(estimated_value))], linestyle="--")

    plt.legend(legends)
    plt.xlabel("Steps")
    plt.ylabel("% Best Arm Pulled")

    plt.title("% Best Arm Pulled") if title is None else plt.title(title)
    plt.savefig(save_name) if save_name is not None else plt.show()
    return None


def train(num_actions: int, num_runs: int, num_steps: int,
          agent: AGENT_CLASS, env: ENV_CLASS,
          agent_info: Dict = None, env_info: Dict = None,
          return_mean_reward: bool = True,
          resample_reward_step: int = None
          ) -> Tuple[np.ndarray[np.float64], float, np.ndarray[np.float64],\
                     np.ndarray[np.float64], np.ndarray[np.float64]]\
        :
    """
    :param num_actions: number of actions/ k-armed bandits
    :param num_runs: number of experiment an agent is going to perform
    :param num_steps: number of step for each conducted experiment
    :param agent: Check AGENT_CLASS for more details
    :param agent_info:  //
    :param env: Check ENV_CLASS for more details
    :param env_info: //
    :param return_mean_reward: take mean reward over all runs/ experiments
    :param resample_reward_step: number of steps to resample reward of k-arm from normal dist
    :return:
        if return_mean:
            mean reward matrix with shape (num_runs)
        else:
            rewards matrix with shape (num_runs, num_steps)
    Basic training loop
    """
    average_best = 0
    expected_value = None
    estimated_value: np.ndarray = np.zeros(shape=(num_actions, num_steps), dtype=np.float64)

    rewards: np.ndarray = np.zeros((num_runs, num_steps), dtype=np.float32)
    best_action_chosen: np.ndarray = np.zeros((num_runs, num_steps), dtype=np.int32)
    print("--- Start training progress ---")

    for run in tqdm(range(num_runs)):
        np.random.seed(1)

        rl_glue = RLGlue(agent, env)
        rl_glue.rl_init(agent_info, env_info)
        rl_glue.rl_start()

        average_best += np.max(rl_glue.environment.arms)
        best_arm = np.argmax(rl_glue.environment.arms)

        # Just take 1st run for visualization reason
        if run == 0:
            expected_value = rl_glue.environment.arms

        for i in range(num_steps):
            reward, _, action, _ = rl_glue.rl_step()

            rewards[run, i] = reward

            if action == best_arm:
                best_action_chosen[run, i] = 1

            # Just take 1st run for visualization reason
            if run == 0:
                # estimated_value[rl_glue.agent.num_actions*i: rl_glue.agent.num_actions*(i+1)] = rl_glue.agent.q_values
                estimated_value[:, i] = rl_glue.agent.q_values

            # Resample reward from normal dist
            if (resample_reward_step is not None) and (i > 0) and (i % resample_reward_step == 0):
                rl_glue.environment.arms = np.random.randn(10)

    best_action_chosen = np.mean(best_action_chosen, axis=0)

    if return_mean_reward:
        rewards = np.mean(rewards, axis=0)
    return rewards, average_best, best_action_chosen, expected_value, estimated_value


def visualize_value_fn(V: np.ndarray, pi: np.ndarray) -> None:
    """
    :param V: value fn, shape (states)
    :param pi: policy fn, shape (states, actions)
    :return: None

    """
    plt.rc('font', size=15)  # controls default text sizes

    # plot value
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
    ax1.axis('on')
    ax1.cla()
    states = np.arange(V.shape[0])
    ax1.bar(states, V, edgecolor='none')
    ax1.set_xlabel('State')
    ax1.set_ylabel('Value', rotation='horizontal', ha='right')
    ax1.set_title('Value Function')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax1.yaxis.grid()
    ax1.set_ylim(bottom=V.min())

    # plot policy
    ax2.axis('on')
    ax2.cla()
    im = ax2.imshow(pi.T, cmap='Greys', vmin=0, vmax=1, aspect='auto')
    ax2.invert_yaxis()
    ax2.set_xlabel('State')
    ax2.set_ylabel('Action', rotation='horizontal', ha='right')
    ax2.set_title('Policy')
    start, end = ax2.get_xlim()
    ax2.xaxis.set_ticks(np.arange(start, end), minor=True)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    start, end = ax2.get_ylim()
    ax2.yaxis.set_ticks(np.arange(start, end), minor=True)
    ax2.grid(which='minor')

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.20)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('Prob', rotation=0, ha='left')
    fig.subplots_adjust(wspace=0.5)

    # plt.savefig("demo.png")
    plt.show()
