import numpy as np


class QLearningAgent:
    """Tabular Q-learning agent for environments with discrete states and actions."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((n_states, n_actions))

    def greedy_action(self, state):
        q = self.q_table[state]
        best_actions = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best_actions))

    def act(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(state)

    def update(self, state, action, reward, next_state, terminated):
        if terminated:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])
