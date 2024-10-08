from tqdm import tqdm
from semester_8.REL301m.REL_lib.agent import FrozenLakeBellmanAgent
from semester_8.REL301m.REL_lib.environment import CustomFrozenLakeENV


def main() -> None:
    env: CustomFrozenLakeENV = CustomFrozenLakeENV(None and "human", None, "8x8", True)
    env.env_start()

    agent = FrozenLakeBellmanAgent(env)
    pi = agent.train()

    tot_rew = 0
    num_games = 1000
    state, _ = env.reset()
    for _ in tqdm(range(num_games), "Testing"):
        term = False
        while not term:
            (reward, next_state, term, _) = env.step(int(pi[state]))
            state = next_state
            tot_rew += reward
            if term:
                state, _ = env.reset()
    print(f"Won {tot_rew} of {num_games} games!\n")
    return None


if __name__ == '__main__':
    main()
