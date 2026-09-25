# Task 9 - Training Visualization

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------
# Load training data
# --------------------------------

data = np.load("outputs/training_data.npz")

episode_rewards = data["episode_rewards"]
episode_steps = data["episode_steps"]
episode_wins = data["episode_wins"]


episodes = np.arange(1, len(episode_rewards) + 1)


# --------------------------------
# Calculate rolling performance
# --------------------------------

window = 100

rolling_win_rate = (
    np.convolve(
        episode_wins,
        np.ones(window) / window,
        mode="valid"
    ) * 100
)

rolling_reward = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)


# --------------------------------
# Plot 1 - Win Rate
# --------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    episodes[window - 1:],
    rolling_win_rate
)

plt.xlabel("Episode")
plt.ylabel("Win Rate (%)")
plt.title("Grid World RL - Win Rate During Training")
plt.grid(True)

plt.savefig(
    "outputs/win_rate_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# --------------------------------
# Plot 2 - Average Reward
# --------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    episodes[window - 1:],
    rolling_reward
)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Grid World RL - Reward During Training")
plt.grid(True)

plt.savefig(
    "outputs/reward_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# --------------------------------
# Plot 3 - Steps per Episode
# --------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    episodes,
    episode_steps,
    alpha=0.4
)

plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Grid World RL - Steps per Episode")
plt.grid(True)

plt.savefig(
    "outputs/steps_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# --------------------------------
# Final Summary
# --------------------------------

print("\nVisualization completed!")

print("\nGenerated files:")

print("1. outputs/win_rate_curve.png")
print("2. outputs/reward_curve.png")
print("3. outputs/steps_curve.png")

print("\nTraining Summary")

print(
    "Overall Win Rate:",
    f"{np.mean(episode_wins) * 100:.2f}%"
)

print(
    "Average Reward:",
    f"{np.mean(episode_rewards):.2f}"
)

print(
    "Average Steps:",
    f"{np.mean(episode_steps):.2f}"
)
