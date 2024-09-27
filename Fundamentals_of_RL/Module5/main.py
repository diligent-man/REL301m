import copy
import numpy as np
from typing import Tuple, Union

from semester_8.REL301m.REL_lib import (
    ParkingWorld,
    visualize_value_fn
)


class Global:
    num_spaces = 10  # 2 spaces but 3 states including 0, 1, 2
    num_prices = 4  # action space
    gamma = 0.9
    theta = 0.1

    # Environment
    env: ParkingWorld = ParkingWorld(num_spaces, num_prices)

    # value fn where the  i-th entry gives the value of i spaces being occupied
    V: np.ndarray = np.zeros(num_spaces + 1)

    # 2D policy that represents the prob of taking action j/ col in state i/ row
    pi: np.ndarray = np.ones((num_spaces+1, num_prices)) / num_prices  # uniform dist


def inspect_visualize_value_fn() -> None:
    V, pi = copy.deepcopy(Global.V), copy.deepcopy(Global.pi)

    # Sampling from normal dist
    for i in range(len(V)):
        V[i] = np.random.rand()

    visualize_value_fn(V, pi)
    return None


def inspect_transition_prob() -> None:
    states = 3
    action = 2
    transitions = Global.env.transitions(states, action)

    for s_prime, (reward, prob) in enumerate(transitions):
        print(f'p(S\'={s_prime}, R={reward} | S={states}, A={action}) = {prob.round(2)}')
    return None


def bellman_update_v1(current_state: int,
                      env: ParkingWorld,
                      V: np.ndarray,
                      gamma: float,
                      pi: np.ndarray = None,
                      weighting: bool = True
                     ) -> float:
    """
    :param current_state
    :param env: ParkingWorld nxn grid env
    :param V: values fn
    :param pi: policy matrix containing actions and their prob in each state (states, actions)
    :param gamma: discount factor
    :param weighting
        If true:
            estimate new V(s) for the given policy.
            ==> Check https://gurpreet-ai.github.io/policy-evaluation-deep-reinforcement-learning-series/
        else:
            optimize Policy pi(s).
            ==> Check https://gurpreet-ai.github.io/policy-iteration-deep-reinforcement-learning-series/
    :return:
        If weighting:
            return weighted V(s)
        else:
            return unweighted V(s)
    """
    """
    for each action a
        action_prob = pi[current_state][action] if weighting else 1
        
        for each env.transitions(s, a) t
            reward, transition_prob = transition[t]
            new_v += pi[s, a] * transition_prob * [reward + gamma * V[t]]
    """
    if weighting:
        assert pi is not None, "Need providing prob of action for weighting state transition prob"

    new_v = 0.
    for a in env.A:  # for all actions
        action_prob = pi[current_state, a] if weighting else 1

        for s_prime, (reward, transition_prob) in enumerate(env.transitions(current_state, a)):  # for all transitions
            new_v += action_prob * transition_prob * (reward + gamma * V[s_prime])
    return new_v


def bellman_update_v2(current_state: int,
                      env: ParkingWorld,
                      V: np.ndarray,
                      gamma: float,
                      pi: np.ndarray = None,
                      weighting: bool = True
                      ) -> Union[float, np.ndarray]:

    """
    :param current_state
    :param env: ParkingWorld nxn grid env
    :param V: values fn
    :param gamma: discount factor
    :param pi: policy matrix containing actions and their prob in each state (states, actions).
               Used for weighting updated_value
    :param weighting
        If true:
            estimate new V(s) for the given policy.
            ==> Check https://gurpreet-ai.github.io/policy-evaluation-deep-reinforcement-learning-series/
        else:
            optimize Policy pi(s).
            ==> Check https://gurpreet-ai.github.io/policy-iteration-deep-reinforcement-learning-series/
    :return: updated_value
    """
    """
    In this version, I utilize matrix multiplication for estimating V(s) or pi(s)
    Formula: A(s) * P(s', r| s, a) * (R(s, a, s') + gamma * V(s')), where
        A(s): action_probs
        P(s', r| s, a): prob of transitioning from s to s'
        R(s, a, s): reward of transitioning from s to s
        V(s'): values fn of s_prime

    Example: The MDP system has 4 states and 3 actions, we have as the following,

    A(s) = [a1 a2 a3] if weighting else np.ones(len(actions))

    P(s', r| s, a) = [[p1    p2     p3     p4],      , where
                      [p1'   p2'    p3',   p4'],
                      [p1''  p2''   p3''   p4'']]
        p1, 2, ...     : prob of transitioning from s to s' when taking action a1
        p1', 2', ...   :                   //                                  a2
        p1'', 2'', ... :                   //                                  a3

    R(s, a, s') = [[r1   r1'  r1''],                , where
                   [r2   r2'  r2''],
                   [r3   r3'  r3''],
                   [r4   r4'  r4'']]
        r1   : coresponding reward when transitioning from s to s' after taking action a1
        r1'  :                              //                                         a2
        r1'' :                              //                                         a3

    V(s') = [[V(s1)  V(s1')  V(s1'')],            , where 
             [V(s2)  V(s2')  V(s2'')],
             [V(s3)  V(s3')  V(s3'')]
             [V(s4)  V(s4')  V(s4'')]]
        V(s1)   : value of next state w.r.t current state when taking action a1 
        V(s1')  :                           //                               s2
        V(s1'') :                           //                               a3


    Shape for matrix multiplication:
    new_value =       A(s)         *    P(s', r| s, a)      *      (R(s, a, s')        +       gamma  *  V(s'))
    (1, num_actions)  (1, num_actions)  (num_actions, num_states)   (num_states, num_actions)  scalar    (num_states, num_actions)

    In the case of optimizing policy, we're gonna take argmax of new_valu. But in this case, I took mean of all action values
    """
    # (num_actions, num_states, 2)
    transitions_mat = np.array([env.transitions(current_state, action) for action in env.A])
    prob = transitions_mat[..., 1]
    reward = transitions_mat[..., 0]

    # (num_states, num_actions)
    s_prime: np.ndarray = (np.arange(len(env.S), dtype=np.int32)).reshape(-1, 1)
    s_prime = np.repeat(s_prime, repeats=len(env.A), axis=1)
    s_prime = np.apply_along_axis(lambda row: [V[ele] for ele in row], axis=0, arr=s_prime)

    # (1, num_actions)
    if weighting:
        # Updated value V(s) = Pi(a|s) * p(s', r|s, a) * (r(s, a, s') + gamma * V(s'))
        assert pi is not None, "Need providing prob of action for weighting state transition prob"
        action_probs = pi[current_state, :]
        action_probs = action_probs.reshape((1, -1))
        updated_val: np.ndarray = action_probs @ prob @ (reward.T + gamma * s_prime)
    else:
        # Updated value pi(s) = p(s', r|s, a) * (r(s, a, s') + gamma * V(s'))
        updated_val = prob @ (reward.T + gamma * s_prime)
    return updated_val[:, 0].squeeze()
########################################################################################################################


def evaluate_policy(env: ParkingWorld,
                    V: np.ndarray,
                    pi: np.ndarray,
                    gamma: float,
                    theta: float,
                    v1: bool = False
                    ) -> np.ndarray:
    """
    :param env: nxm grid
    :param V: values fn
    :param pi: policy matrix containing actions and their prob in each state (states, actions)
    :param s: current state
    :param gamma: discount factor
    :param theta: threshold for stop estimating
    :param v1: update with bellman v1 or not
    :return: updated V
    """
    """
    Iterative evaluation for estimating V ~ v_pi 
        delta = +inf
        
        Loop until delta > theta 
            for each state s:
                current_v,  = V[s], 0. 
                V[s] = bellman_update(current_v, V, pi, gamma, weighting=True)
                perform thresholding
    """
    delta = float("inf")
    while delta > theta:
        delta = 0

        for s in env.S:
            current_v, new_v = V[s], 0.
            V[s] = bellman_update_v1(s, env, V, gamma, pi) if v1 else bellman_update_v2(s, env, V, gamma, pi)
            delta = np.max((delta, np.abs(current_v - V[s])))  # Thresholding
    return V


def improve_policy(env: ParkingWorld,
                   V: np.ndarray,
                   pi: np.ndarray,
                   gamma: float,
                   v1: bool = False
                   ) -> Tuple[np.ndarray, bool]:
    policy_stable = True

    for s in env.S:
        current_action = pi[s, :].copy()
        new_best_action = bellman_update_v1(s, env, V, gamma, weighting=False) if v1 else bellman_update_v2(s, env, V, gamma, weighting=False)
        new_best_action = np.argmax(new_best_action)

        pi[s, :] = 0
        pi[s, new_best_action] = 1

        # check policy is improved or not
        if not np.array_equal(pi[s, :], current_action):
            policy_stable = False
    return pi, policy_stable


def policy_iteration(env: ParkingWorld,
                     V: np.ndarray,
                     pi: np.ndarray,
                     gamma: float, theta: float,
                     v1: bool = False
                     ) -> Tuple[np.ndarray, np.ndarray]:
    policy_stable = False

    while not policy_stable:
        V = evaluate_policy(env, V, pi, gamma, theta, v1)
        pi, policy_stable = improve_policy(env, V, pi, gamma, v1)
    return V, pi
########################################################################################################################


def value_iteration_v1(env: ParkingWorld, gamma: float, theta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    In this version, value is updated in the similar fashion to evaluate_policy but with max estimated value for faster
    convergence.
    """
    V = np.zeros(len(env.S))
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)

    while True:
        delta = 0
        for s in env.S:
            current_value = V[s]
            V[s] = np.max(bellman_update_v2(s, env, V, gamma, weighting=False))
            delta = max(delta, abs(current_value - V[s]))
        if delta < theta:
            break

    pi, _ = improve_policy(env, V, pi, gamma)
    return V, pi


def value_iteration_v2(env: ParkingWorld, gamma: float, theta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    In this version, nothing but policy is improved simultaneously with value
    """
    V = np.zeros(len(env.S))
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)

    while True:
        delta = 0
        for s in env.S:
            current_value = V[s]
            pi, _ = improve_policy(env, V, pi, gamma)
            V[s] = np.max(bellman_update_v2(s, env, V, gamma, weighting=False))
            delta = max(delta, abs(current_value - V[s]))
        if delta < theta:
            break
    return V, pi


def main() -> None:
    # inspect_visualize_value_fn()
    # inspect_transition_prob()

    V1, pi1 = policy_iteration(Global.env, Global.V.copy(), Global.pi.copy(), Global.gamma, Global.theta, v1=False)
    V2, pi2 = policy_iteration(Global.env, Global.V.copy(), Global.pi.copy(), Global.gamma, Global.theta, v1=False)
    visualize_value_fn(V1, pi1)
    visualize_value_fn(V2, pi2)
    print(np.array_equal(V1, V2))
    print(np.array_equal(pi1, pi2))

    V, pi = value_iteration_v2(Global.env, Global.gamma, Global.theta)
    visualize_value_fn(V, pi)
    return None


if __name__ == '__main__':
    main()
