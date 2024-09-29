import numpy as np

from semester_8.REL301m.REL_lib.agent import (
    GreedyKArmAgent,
    EpsilonGreedyKArmAgent,
    EpsilonGreedyKArmAgentConstantStepsize
)


def test_greedy_agent() -> None:
    np.random.seed(1)
    greedyAgent = GreedyKArmAgent()
    agent_info = {
        "values": np.array([0, 0, 1.0, 0, 0]),
        "arm_count": np.array([0, 1, 0, 0, 0]),
        "last_action": 1,

    }
    greedyAgent.agent_init(agent_info)

    # take a fake agent step
    action = greedyAgent.agent_step(reward=1, obs=)
    assert action == 2
    assert greedyAgent.values.tolist() == [0, 0.5, 1.0, 0, 0]

    # take another step
    action = greedyAgent.agent_step(reward=2, obs=)
    assert action == 2
    assert greedyAgent.values.tolist() == [0, 0.5, 2.0, 0, 0]
    print("All tests passed!")
    return None


def test_epsilon_greedy_agent() -> None:
    np.random.seed(0)
    epsilonGreedyAgent = EpsilonGreedyKArmAgent()
    agent_info = {
        "values": np.array([0, 0, 1.0, 0, 0]),
        "arm_count": np.array([0, 1, 0, 0, 0]),
        "num_actions": 5,
        "last_action": 1,
        "epsilon": 0.5
    }
    epsilonGreedyAgent.agent_init(agent_info)

    epsilonGreedyAgent.agent_step(reward=1, obs=)
    assert epsilonGreedyAgent.values.tolist() == [0, 0.5, 1.0, 0, 0]

    # manipulate the random seed so the agent takes a random action
    np.random.seed(1)
    action = epsilonGreedyAgent.agent_step(reward=0, obs=)
    assert action == 4

    # check to make sure we update value for action 4
    epsilonGreedyAgent.agent_step(reward=1, obs=)
    assert epsilonGreedyAgent.values.tolist() == [0, 0.5, 0.0, 0, 1.0]
    print("All tests passed!")
    return None


def test_epsilon_greedy_agent_constant_stepsize() -> None:
    np.random.seed(0)
    epsilonGreedyAgent = EpsilonGreedyKArmAgentConstantStepsize()

    for step_size in [0.01, 0.1, 0.5, 1.0]:
        agent_info = {
            "values": np.array([0, 0, 1.0, 0, 0]),
            "num_actions": 5,
            "last_action": 1,
            "epsilon": 0.,
            "step_size": step_size
        }
        epsilonGreedyAgent.agent_init(agent_info)

        epsilonGreedyAgent.agent_step(reward=1, obs=)
        assert epsilonGreedyAgent.values.tolist() == [0, step_size, 1.0, 0, 0]
    print("All tests passed!")
    return None


def main() -> None:
    # Your code
    test_greedy_agent()
    test_epsilon_greedy_agent()
    test_epsilon_greedy_agent_constant_stepsize()
    return None


if __name__ == '__main__':
    main()