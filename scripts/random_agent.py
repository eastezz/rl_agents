import gymnasium as gym

ACTION_NAMES = ["left", "down", "right", "up"]


def main():
    env = gym.make("FrozenLake-v1", is_slippery=True)
    state, info = env.reset(seed=42)
    print(f"start state: {state}")

    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        step += 1
        print(
            f"step {step:3d} | state {state:2d} | action {action} ({ACTION_NAMES[action]:5s}) "
            f"| next {next_state:2d} | reward {reward} | terminated {terminated}"
        )
        state = next_state

    print("reached the goal!" if reward == 1 else "fell in a hole or ran out of steps")
    env.close()


if __name__ == "__main__":
    main()
