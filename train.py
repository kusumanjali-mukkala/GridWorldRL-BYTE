# Task 9 - Q-Learning Training

import numpy as np

from environment import GridWorld
from agent import QLearningAgent


# -----------------------------
# Training settings
# -----------------------------

EPISODES = 5000
MAX_STEPS = 100


# -----------------------------
# Create environment and agent
# -----------------------------

environment = GridWorld()

agent = QLearningAgent(
    rows=5,
    columns=5,
    actions=4
)


# -----------------------------
# Store training results
# -----------------------------

episode_rewards = []
episode_steps = []
episode_wins = []


# -----------------------------
# Training loop
# -----------------------------

for episode in range(EPISODES):

    state = environment.reset()

    total_reward = 0
    steps = 0
    done = False

    for step in range(MAX_STEPS):

        # Choose an action
        action = agent.choose_action(state)

        # Perform the action
        next_state, reward, done = environment.step(action)

        # Update Q-table
        agent.update_q_value(
            state,
            action,
            reward,
            next_state,
            done
        )

        # Move to next state
        state = next_state

        total_reward += reward
        steps += 1

        # Stop if goal or trap is reached
        if done:
            break

    # Reduce exploration gradually
    agent.decay_epsilon()

    # Save episode results
    episode_rewards.append(total_reward)
    episode_steps.append(steps)

    # Check whether agent reached the goal
    if done and reward == 100:
        episode_wins.append(1)
    else:
        episode_wins.append(0)


    # -----------------------------
    # Print progress
    # -----------------------------

    if (episode + 1) % 500 == 0:

        win_rate = np.mean(
            episode_wins[-500:]
        ) * 100

        average_reward = np.mean(
            episode_rewards[-500:]
        )

        print(
            f"Episode: {episode + 1} | "
            f"Win Rate: {win_rate:.2f}% | "
            f"Average Reward: {average_reward:.2f} | "
            f"Epsilon: {agent.epsilon:.4f}"
        )


# -----------------------------
# Save Q-table
# -----------------------------

agent.save_q_table(
    "models/q_table.npy"
)


# -----------------------------
# Save training history
# -----------------------------

np.savez(
    "outputs/training_data.npz",
    episode_rewards=np.array(episode_rewards),
    episode_steps=np.array(episode_steps),
    episode_wins=np.array(episode_wins)
)


# -----------------------------
# Final results
# -----------------------------

print("\nTraining completed!")

print(
    "Final epsilon:",
    agent.epsilon
)

print(
    "Overall win rate:",
    f"{np.mean(episode_wins) * 100:.2f}%"
)

print(
    "Average reward:",
    f"{np.mean(episode_rewards):.2f}"
)

print(
    "\nQ-table saved to:",
    "models/q_table.npy"
)

print(
    "Training data saved to:",
    "outputs/training_data.npz"
)