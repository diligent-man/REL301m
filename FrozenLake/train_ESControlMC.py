import sys
sys.path.append("/home/trong/Downloads/Local/Source/Python")

from tqdm import tqdm
from semester_8.REL301m.REL_lib.agent import FrozenLakeESControlMCAgent
from semester_8.REL301m.REL_lib.environment import CustomFrozenLakeENV


def main() -> None:
    env: CustomFrozenLakeENV = CustomFrozenLakeENV(None and "human", None, "8x8", True)
    env.env_start()

    agent = FrozenLakeESControlMCAgent(env, eval_episodes=10000, impr_episodes=10000)
    pi = agent.train(num_iters=1000)
    print(pi)

    tot_rew = 0
    num_games = 1000
    current_step = 0
    truncation_step = 1000
    state, _ = env.reset()
    for _ in tqdm(range(num_games), "Testing"):
        term = False

        while not term and current_step <= truncation_step:
            (reward, next_state, term, _) = env.step(int(pi[state]))
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