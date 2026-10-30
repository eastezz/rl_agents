import numpy as np
import pytest

from rl_agents.plot import moving_average, plot_learning_curve, print_policy


def test_moving_average():
    values = np.array([0.0, 0.0, 1.0, 1.0])
    assert moving_average(values, window=2) == pytest.approx([0.0, 0.5, 1.0])


def test_plot_learning_curve_saves_a_png(tmp_path):
    successes = np.array([False, True] * 100)
    out_file = tmp_path / "curve.png"
    plot_learning_curve(successes, window=10, title="test run", out_file=out_file)
    assert out_file.stat().st_size > 0


def test_print_policy_draws_the_frozenlake_map(capsys):
    q_table = np.zeros((16, 4))
    q_table[:, 2] = 1.0  # "right" is the best action on every tile
    print_policy(q_table, "FrozenLake-v1")
    assert capsys.readouterr().out.splitlines() == [
        "→ → → →",
        "→ H → H",
        "→ → → H",
        "H → → G",
    ]


def test_print_policy_skips_taxi(capsys):
    """Regression test: Taxi also has a map, but its 500 states are not tiles. This used to crash."""
    print_policy(np.zeros((500, 6)), "Taxi-v4")
    assert "only available for grid environments" in capsys.readouterr().out
