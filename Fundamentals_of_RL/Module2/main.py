import os
import numpy as np

from typing import List
from semester_8.REL301m.REL_lib import (
    GreedyKArmAgent,
    EpsilonGreedyKArmAgent,
    EpsilonGreedyKArmAgentConstantStepsize,

    TenArmEnv,

    visualize_training_result,
    visualize_best_action_chosen,
    visualize_step_size_effect_to_q_value,
    k_armed_bandit_train
)


class Global:
    num_actions = 10
    num_runs = 20000
    num_steps = 2000

    agent_1_info = {"num_actions": num_actions}
    agent_2_info = {"num_actions": num_actions, "epsilon": 0.01}
    agent_3_info = {"num_actions": num_actions, "epsilon": 0.1}
    agent_4_info = {"num_actions": num_actions, "epsilon": 0.5}


def exploration_exploitation_trade_off():
    reward_lst = None
    legends: List[str] = []
    avg_best_lst: List[float] = []

    for i, (agent, env, agent_info) in enumerate(zip(
            [*[GreedyKArmAgent()], *[EpsilonGreedyKArmAgent()] * 3],
            [TenArmEnv()] * 4,
            (Global.agent_1_info, Global.agent_2_info, Global.agent_3_info, Global.agent_4_info)
    )):
        if agent.__class__.__name__ == "GreedyAgent":
            legends.insert(i, "Greedy Avg Best")
            legends.append("Greedy Agent")

        elif agent.__class__.__name__ == "EpsilonGreedyAgent":
            legends.insert(i, f"Eps Greedy Avg Best {agent_info['epsilon']}")
            legends.append(f"Eps Greedy Agent {agent_info['epsilon']}")

        mean_rewards, avg_best, _, _, _ = k_armed_bandit_train(Global.num_actions, Global.num_runs, Global.num_steps, agent, env, agent_info)

        reward_lst = mean_rewards if reward_lst is None else np.vstack((reward_lst, mean_rewards))
        avg_best_lst.append(avg_best)

    visualize_training_result(Global.num_runs, Global.num_steps,
                              reward_lst, avg_best_lst,
                              legends,
                              "Exploration-Exploitation trade-off",
                              "exploration_eploitation_trade_off.png"
                              )


def step_size_effect_to_chosen_action() -> None:
    step_sizes = [0.01, 0.1, 0.5, 1.0, "1/N(A)"]

    avg_best_lst: List[float] = []
    reward_lst = best_action_lst = None

    training_legends: List[str] = []
    best_action_legends: List[str] = []

    for i in range(len(step_sizes)):
        agent_info = Global.agent_1_info

        if step_sizes[i] == "1/N(A)":
            agent = EpsilonGreedyKArmAgent()
            training_legends.insert(i, f"Avg best ss=1/N(A)")
            training_legends.append(f"Greedy Agent ss=1/N(A)")

        else:
            agent = EpsilonGreedyKArmAgentConstantStepsize()
            agent_info = {**agent_info, "step_size": step_sizes[i]}

            training_legends.insert(i, f"Avg best ss={step_sizes[i]}")
            training_legends.append(f"Greedy Agent ss={step_sizes[i]}")

        env = TenArmEnv()
        mean_rewards, avg_best, best_action_chosen, _, _ = k_armed_bandit_train(Global.num_actions, Global.num_runs, Global.num_steps, agent, env, agent_info)

        # Update tracked list
        avg_best_lst.append(avg_best)
        reward_lst = mean_rewards if reward_lst is None else np.vstack((reward_lst, mean_rewards))
        best_action_lst = best_action_chosen if best_action_lst is None else np.vstack((best_action_lst, best_action_chosen))
        best_action_legends.append(f"Step size {step_sizes[i]}")

    visualize_training_result(Global.num_runs, Global.num_steps,
                              reward_lst, avg_best_lst,
                              training_legends,
                              "Effect of step size",
                              "step_size_effect.png"
    )

    visualize_best_action_chosen(best_action_lst, best_action_legends, save_name="step_size_effect_to_best_action_chosen.png")
    return None


def step_size_effect_to_estimated_q_value() -> None:
    os.makedirs("step_size_effect", exist_ok=True)

    step_sizes = [0.01, 0.1, 0.5, 1.0, "1_N(A)"]
    for i in range(len(step_sizes)):
        agent_info = Global.agent_3_info

        if step_sizes[i] == "1_N(A)":
            agent = EpsilonGreedyKArmAgent()
        else:
            agent = EpsilonGreedyKArmAgentConstantStepsize()
            agent_info = {**agent_info, "step_size": step_sizes[i]}

        env = TenArmEnv()
        mean_rewards, _, _, expected_value, estimated_value = k_armed_bandit_train(Global.num_actions, Global.num_runs, Global.num_steps, agent, env, agent_info)

        max_expected_value: np.ndarray = np.argmax(expected_value)
        estimated_value = estimated_value[max_expected_value, :]

        visualize_step_size_effect_to_q_value(max_expected_value,
                                              estimated_value,
                                              ["Expected value", "Estimated value"],
                                              f"Step size {step_sizes[i]}",
                                              os.path.join("step_size_effect", f"step_size_{step_sizes[i]}.png"))
    return None


def step_size_and_env_effect_to_reward() -> None:
    reward_lst = None
    legends: List[str] = []
    avg_best_lst: List[float] = []

    step_sizes = [0.01, 0.1, 0.5, 1.0, "1_N(A)"]
    for i in range(len(step_sizes)):
        agent_info = Global.agent_3_info

        if step_sizes[i] == "1_N(A)":
            agent = EpsilonGreedyKArmAgent()
            legends.insert(i, f"Avg best ss=1/N(A)")
            legends.append(f"Greedy Agent ss=1/N(A)")

        else:
            agent = EpsilonGreedyKArmAgentConstantStepsize()
            agent_info = {**agent_info, "step_size": step_sizes[i]}

            legends.insert(i, f"Avg best ss={step_sizes[i]}")
            legends.append(f"Greedy Agent ss={step_sizes[i]}")

        env = TenArmEnv()
        mean_rewards, avg_best, _, _, _ = k_armed_bandit_train(Global.num_actions, Global.num_runs, Global.num_steps,
                                                agent, env, agent_info,
                                                resample_reward_step=500)

        reward_lst = mean_rewards if reward_lst is None else np.vstack((reward_lst, mean_rewards))
        avg_best_lst.append(avg_best)

    visualize_training_result(Global.num_runs, Global.num_steps,
                              reward_lst, avg_best_lst,
                              legends,
                              "Step size (ss) & Env effect to reward",
                              "step_size_and_env_effect_to_reward.png")
    return None


def main() -> None:
    # exploration_exploitation_trade_off()
    # step_size_effect_to_chosen_action()
    # step_size_effect_to_estimated_q_value()
    # step_size_and_env_effect_to_reward()
    return None


if __name__ == '__main__':
    main()