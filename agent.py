# Task 9 - Grid World Q-Learning Agent

import numpy as np


class QLearningAgent:

    def __init__(
        self,
        rows,
        columns,
        actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    ):

        # Grid information
        self.rows = rows
        self.columns = columns
        self.actions = actions

        # Q-learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Exploration parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Create Q-table
        self.q_table = np.zeros(
            (rows, columns, actions)
        )

    def choose_action(self, state):

        row, column = state

        # Exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(self.actions)

        # Exploitation
        return np.argmax(self.q_table[row, column])

    def update_q_value(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        row, column = state
        next_row, next_column = next_state

        current_q = self.q_table[row, column, action]

        # If episode has ended, there is no future Q-value
        if done:
            target = reward

        else:
            best_next_q = np.max(
                self.q_table[next_row, next_column]
            )

            target = (
                reward
                + self.discount_factor * best_next_q
            )

        # Q-learning update
        self.q_table[row, column, action] = (
            current_q
            + self.learning_rate
            * (target - current_q)
        )

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def save_q_table(self, filename="models/q_table.npy"):

        np.save(filename, self.q_table)

    def load_q_table(self, filename="models/q_table.npy"):

        self.q_table = np.load(filename)


# --------------------------------------------------
# Testing the Q-Learning Agent
# --------------------------------------------------

if __name__ == "__main__":

    agent = QLearningAgent(
        rows=5,
        columns=5,
        actions=4
    )

    print("Q-Table shape:", agent.q_table.shape)

    print("\nInitial Q-Table:")
    print(agent.q_table)

    # Test action selection
    state = (0, 0)

    action = agent.choose_action(state)

    print("\nTesting action selection:")
    print("State:", state)
    print("Selected action:", action)

    # Test Q-value update
    next_state = (0, 1)
    reward = -1
    done = False

    agent.update_q_value(
        state,
        action,
        reward,
        next_state,
        done
    )

    print("\nTesting Q-value update:")
    print("Updated Q-value:",
          agent.q_table[state[0], state[1], action])

    # Test epsilon decay
    print("\nEpsilon before decay:", agent.epsilon)

    agent.decay_epsilon()

    print("Epsilon after decay:", agent.epsilon)