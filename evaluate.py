# Task 9 - Grid World Evaluation

import numpy as np

from environment import GridWorld
from agent import QLearningAgent


# -----------------------------------------
# Create environment
# -----------------------------------------

environment = GridWorld()


# -----------------------------------------
# Create Q-learning agent
# -----------------------------------------

agent = QLearningAgent(
    rows=5,
    columns=5,
    actions=4
)


# -----------------------------------------
# Load trained Q-table
# -----------------------------------------

agent.load_q_table("models/q_table.npy")


# -----------------------------------------
# Evaluation settings
# -----------------------------------------

MAX_STEPS = 100

# No exploration during evaluation
agent.epsilon = 0.0


# -----------------------------------------
# Start evaluation
# -----------------------------------------

state = environment.reset()

path = [state]

total_reward = 0
done = False


print("Starting evaluation...")
print("Start position:", state)


# -----------------------------------------
# Follow learned policy
# -----------------------------------------

for step in range(MAX_STEPS):

    # Choose the best learned action
    action = agent.choose_action(state)

    # Perform action
    next_state, reward, done = environment.step(action)

    # Store path
    path.append(next_state)

    # Add reward
    total_reward += reward

    # Move to next state
    state = next_state

    # Stop if goal or trap is reached
    if done:
        break


# -----------------------------------------
# Calculate results
# -----------------------------------------

steps_taken = len(path) - 1

goal_reached = (
    done
    and reward == 100
)


# -----------------------------------------
# Calculate path efficiency
# -----------------------------------------

# Shortest successful path for this grid
optimal_steps = 8

if goal_reached:
    path_efficiency = (
        optimal_steps / steps_taken
    ) * 100
else:
    path_efficiency = 0


# -----------------------------------------
# Display results
# -----------------------------------------

print("\nEvaluation Results")
print("------------------")

print("Goal reached:", goal_reached)

print("Steps taken:", steps_taken)

print("Total reward:", total_reward)

print("Optimal steps:", optimal_steps)

print(
    "Path efficiency:",
    f"{path_efficiency:.2f}%"
)


# -----------------------------------------
# Display learned path
# -----------------------------------------

print("\nLearned Path:")

for position in path:
    print(position)


# -----------------------------------------
# Display final position
# -----------------------------------------

print("\nFinal position:", state)

if goal_reached:
    print("Result: SUCCESS - Goal reached!")
else:
    print("Result: FAILED - Goal not reached.")