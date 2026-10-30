import numpy as np
import pytest

from rl_agents.train import epsilon_schedule, evaluate, train


@pytest.mark.parametrize(
    ("episode", "expected"),
    [
        (0, 1.0),  # start: always explore
        (50, 0.505),  # halfway through the decay
        (100, 0.01),  # end of the decay
        (150, 0.01),  # after the decay it stays at the minimum
    ],
)
def test_epsilon_schedule(episode, expected):
    assert epsilon_schedule(episode, decay_episodes=100) == pytest.approx(expected)


def test_train_returns_one_result_per_episode():
    agent, rewards, successes = train("FrozenLake-v1", episodes=50, seed=0)
    assert agent.q_table.shape == (16, 4)
    assert len(rewards) == 50
    assert len(successes) == 50


def test_training_is_reproducible():
    first, _, _ = train("FrozenLake-v1", episodes=200, seed=7)
    second, _, _ = train("FrozenLake-v1", episodes=200, seed=7)
    assert np.array_equal(first.q_table, second.q_table)


def test_trained_agent_beats_random_play():
    """End-to-end: after short training the agent wins far more often than random play (about 1.5%)."""
    agent, _, _ = train("FrozenLake-v1", episodes=3000, seed=42)
    success_rate = evaluate(agent, "FrozenLake-v1", episodes=500, seed=10**6)
    assert success_rate >= 0.5
