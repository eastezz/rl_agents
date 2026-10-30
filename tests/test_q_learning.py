import numpy as np
import pytest

from rl_agents import QLearningAgent


@pytest.fixture
def agent():
    """A fresh agent the size of FrozenLake: 16 states, 4 actions."""
    return QLearningAgent(n_states=16, n_actions=4, seed=0)


def test_q_table_starts_at_zero(agent):
    assert agent.q_table.shape == (16, 4)
    assert np.all(agent.q_table == 0)


def test_update_on_terminal_step_uses_only_the_reward(agent):
    agent.q_table[15] = 5.0  # must be ignored, because the game is over
    agent.update(state=14, action=2, reward=1.0, next_state=15, terminated=True)
    assert agent.q_table[14, 2] == pytest.approx(0.1)  # 0 + 0.1 * (1.0 - 0)


def test_update_on_normal_step_adds_discounted_future(agent):
    agent.q_table[1] = [0.0, 0.5, 2.0, 0.0]  # best value in the next state: 2.0
    agent.update(state=0, action=2, reward=0.0, next_state=1, terminated=False)
    assert agent.q_table[0, 2] == pytest.approx(0.1 * 0.99 * 2.0)


def test_greedy_action_picks_the_best_action(agent):
    agent.q_table[3] = [0.1, 0.9, 0.2, 0.3]
    assert agent.greedy_action(3) == 1


def test_greedy_action_breaks_ties_at_random(agent):
    agent.q_table[0] = [1.0, 0.0, 1.0, 0.0]  # actions 0 and 2 are tied
    chosen = {agent.greedy_action(0) for _ in range(100)}
    assert chosen == {0, 2}


def test_act_without_exploration_is_greedy(agent):
    agent.epsilon = 0.0
    agent.q_table[5] = [0.0, 0.0, 0.0, 1.0]
    assert all(agent.act(5) == 3 for _ in range(50))


def test_act_with_full_exploration_tries_every_action(agent):
    agent.epsilon = 1.0
    chosen = {agent.act(0) for _ in range(200)}
    assert chosen == {0, 1, 2, 3}


def test_same_seed_gives_same_actions():
    first = QLearningAgent(n_states=16, n_actions=4, seed=123)
    second = QLearningAgent(n_states=16, n_actions=4, seed=123)
    assert [first.act(0) for _ in range(20)] == [second.act(0) for _ in range(20)]
