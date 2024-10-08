import sys
sys.path.append("/home/trong/Downloads/Local/Source/Python")

import numpy as np

from tqdm import tqdm
from semester_8.REL301m.REL_lib.agent import FrozenLakeQLearningAgent
from semester_8.REL301m.REL_lib.environment import CustomFrozenLakeENV


def main() -> None:
    env: CustomFrozenLakeENV = CustomFrozenLakeENV(None and "human", None, "8x8", True)
    env.env_start()

    agent: FrozenLakeQLearningAgent = FrozenLakeQLearningAgent(env, eval_episodes=10000, impr_episodes=10000)
    Q: np.ndarray[np.float32] = agent.train(num_iters=1000)

    tot_rew = 0
    num_games = 1000
    current_step = 0
    truncation_step = 1000
    state, _ = env.reset()
    for _ in tqdm(range(num_games), "Testing"):
        term = False

        while not term and current_step <= truncation_step:
            action = np.random.choice(np.flatnonzero(Q[state, :] == np.max(Q[state, :])))
            reward, next_state, term, _ = env.step(action)

            state = next_state
            tot_rew += reward

            if term:
                state, _ = env.reset()
                current_step = 0
            else:
                current_step += 1
    print(f"Won {tot_rew} of {num_games} games!\n")
    return None

if __name__ == '__main__':
    main()