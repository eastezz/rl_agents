import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

ARROWS = ["←", "↓", "→", "↑"]
RANDOM_BASELINE = 0.015


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a learning curve and show the learned policy.")
    parser.add_argument("results", help="path to a .npz file saved by rl_agents.train")
    parser.add_argument("--window", type=int, default=100, help="moving-average window in episodes")
    return parser.parse_args()


def moving_average(values, window):
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_learning_curve(successes, window, title, out_file):
    curve = moving_average(successes.astype(float), window)
    episodes = np.arange(window, len(successes) + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(episodes, curve, color="#2a78d6", linewidth=1.5)
    ax.axhline(RANDOM_BASELINE, color="#52514e", linewidth=1, linestyle="--")
    ax.text(episodes[-1], RANDOM_BASELINE + 0.02, "random agent", color="#52514e", ha="right")

    ax.set_title(title, loc="left")
    ax.set_xlabel("episode")
    ax.set_ylabel(f"success rate (last {window} episodes)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", color="#e5e4e0")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def print_policy(q_table, env_id):
    """Print the greedy action for every tile of a grid map (FrozenLake-style envs)."""
    env = gym.make(env_id)
    desc = getattr(env.unwrapped, "desc", None)
    env.close()
    if desc is None:
        print("(policy map only available for grid environments like FrozenLake)")
        return

    n_rows, n_cols = desc.shape
    for row in range(n_rows):
        cells = []
        for col in range(n_cols):
            tile = desc[row, col].decode()
            state = row * n_cols + col
            cells.append(tile if tile in "HG" else ARROWS[np.argmax(q_table[state])])
        print(" ".join(cells))


def main():
    args = parse_args()
    results_path = Path(args.results)
    data = np.load(results_path)
    env_id = results_path.stem.split("_seed")[0]

    out_file = results_path.with_suffix(".png")
    title = f"Q-learning on {env_id}: greedy test success {float(data['eval_rate']):.1%}"
    plot_learning_curve(data["successes"], args.window, title, out_file)
    print(f"saved learning curve to {out_file}\n")

    print("learned policy (H = hole, G = goal):")
    print_policy(data["q_table"], env_id)


if __name__ == "__main__":
    main()
