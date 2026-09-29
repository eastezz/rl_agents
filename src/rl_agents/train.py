import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

from rl_agents import QLearningAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tabular Q-learning agent.")
    parser.add_argument("--env", default="FrozenLake-v1", help="gymnasium environment id")
    parser.add_argument("--episodes", type=int, default=20000, help="number of training episodes")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--out", default="results", help="folder to save the Q-table and learning curve data")
    return parser.parse_args()


def epsilon_schedule(episode, decay_episodes, eps_start=1.0, eps_end=0.01):
    """Linear decay from eps_start to eps_end over decay_episodes, then stay at eps_end."""
    fraction = min(episode / decay_episodes, 1.0)
    return eps_start + fraction * (eps_end - eps_start)


def train(env_id, episodes, seed, log_every=1000):
    env = gym.make(env_id)
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        seed=seed,
    )
    decay_episodes = int(episodes * 0.8)

    episode_rewards = []
    successes = []
    for episode in range(episodes):
        agent.epsilon = epsilon_schedule(episode, decay_episodes)
        state, info = env.reset(seed=seed + episode)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.update(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        successes.append(terminated and reward > 0)

        if (episode + 1) % log_every == 0:
            success_rate = np.mean(successes[-100:])
            avg_reward = np.mean(episode_rewards[-100:])
            print(
                f"episode {episode + 1:6d} | epsilon {agent.epsilon:.3f} "
                f"| success rate (last 100) {success_rate:6.1%} | avg reward {avg_reward:7.2f}"
            )

    env.close()
    return agent, episode_rewards, successes


def evaluate(agent, env_id, episodes, seed):
    """Run the greedy policy (no exploration, no learning) and return the success rate."""
    env = gym.make(env_id)
    wins = 0
    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)
        terminated = truncated = False
        while not (terminated or truncated):
            state, reward, terminated, truncated, info = env.step(agent.greedy_action(state))
        wins += terminated and reward > 0
    env.close()
    return wins / episodes


def main():
    args = parse_args()
    agent, episode_rewards, successes = train(args.env, args.episodes, args.seed)
    eval_rate = evaluate(agent, args.env, episodes=1000, seed=args.seed + 10**6)
    print(f"\ngreedy policy success rate over 1000 test episodes: {eval_rate:.1%}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.env}_seed{args.seed}.npz"
    np.savez(
        out_file,
        q_table=agent.q_table,
        rewards=np.array(episode_rewards),
        successes=np.array(successes),
        eval_rate=eval_rate,
    )
    print(f"saved results to {out_file}")


if __name__ == "__main__":
    main()
