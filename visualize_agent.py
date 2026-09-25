# Task 9 - Trained Agent GIF Visualization

import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from environment import GridWorld
from agent import QLearningAgent


# --------------------------------
# Create environment and agent
# --------------------------------

environment = GridWorld()

agent = QLearningAgent(
    rows=5,
    columns=5,
    actions=4
)

# Load trained Q-table
agent.load_q_table("models/q_table.npy")

# Use only learned actions
agent.epsilon = 0.0


# --------------------------------
# Grid drawing function
# --------------------------------

def draw_grid(agent_position):

    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw grid
    for row in range(5):
        for column in range(5):

            cell = environment.grid[row][column]

            if cell == "S":
                label = "START"
            elif cell == "G":
                label = "GOAL"
            elif cell == "X":
                label = "TRAP"
            elif cell == "#":
                label = "WALL"
            else:
                label = ""

            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=10
            )

    # Draw agent
    agent_row, agent_column = agent_position

    ax.plot(
        agent_column,
        agent_row,
        marker="o",
        markersize=18
    )

    # Grid settings
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(4.5, -0.5)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    ax.set_title(
        f"Trained RL Agent - Position {agent_position}"
    )

    ax.grid(True)

    return fig


# --------------------------------
# Run trained agent
# --------------------------------

state = environment.reset()

frames = []

path = [state]

done = False

MAX_STEPS = 100


for step in range(MAX_STEPS):

    action = agent.choose_action(state)

    next_state, reward, done = environment.step(action)

    state = next_state

    path.append(state)

    # Create frame
    figure = draw_grid(state)

    figure.canvas.draw()

    width, height = figure.canvas.get_width_height()

    frame = figure.canvas.buffer_rgba()

    frame = frame.__class__(
        frame
    )

    import numpy as np

    frame = np.asarray(
        figure.canvas.buffer_rgba()
    ).copy()

    frames.append(frame)

    plt.close(figure)

    if done:
        break


# --------------------------------
# Save GIF
# --------------------------------

os.makedirs("outputs", exist_ok=True)

gif_path = "outputs/trained_agent.gif"

imageio.mimsave(
    gif_path,
    frames,
    duration=0.6,
    loop=0
)


# --------------------------------
# Print results
# --------------------------------

print("\nAgent visualization completed!")

print("GIF saved to:")
print(gif_path)

print("\nLearned path:")

for position in path:
    print(position)

print("\nTotal steps:", len(path) - 1)

if done and reward == 100:
    print("Result: SUCCESS - Goal reached!")
else:
    print("Result: FAILED - Goal not reached.")