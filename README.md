# Gladiator - Grid World Reinforcement Learning Agent

## Project Overview

This project implements a custom Grid World environment and trains a Reinforcement Learning agent using the Q-Learning algorithm.

The agent learns how to move from a starting position to a goal while avoiding walls and traps.

## Task

Task 9 - Gladiator: Grid World RL Agent

## Objective

The main objective is to train an RL agent that can learn an efficient path from the starting position to the goal.

The agent receives rewards and penalties based on its actions and gradually learns the best actions for different states.

## Environment

The environment is a custom 5 × 5 grid.

Symbols used:

- `S` → Start
- `G` → Goal
- `X` → Trap
- `#` → Wall
- `.` → Empty cell

Starting position:

`(0, 0)`

Goal position:

`(4, 4)`

## Actions

The agent can perform four actions:

- `UP`
- `DOWN`
- `LEFT`
- `RIGHT`

## Reward System

| Situation | Reward |
|---|---:|
| Normal movement | -1 |
| Wall / boundary | -10 |
| Trap | -100 |
| Goal | +100 |

The reward system encourages the agent to reach the goal while avoiding traps and unnecessary movements.

## Algorithm

The project uses **Tabular Q-Learning**.

The agent maintains a Q-table containing the expected value of taking each action from each state.

The agent uses an epsilon-greedy strategy:

- Exploration → try different actions
- Exploitation → choose the action with the highest Q-value

## Training

The agent was trained for:

**5,000 episodes**

Training parameters include:

- Learning rate: `0.1`
- Discount factor: `0.95`
- Initial epsilon: `1.0`
- Minimum epsilon: `0.01`
- Epsilon decay: `0.995`

## Results

Overall Win Rate:

**97.38%**

Average Reward:

**84.06**

Average Steps:

**8.72**

## Evaluation

After training, the agent successfully reached the goal.

Learned path:

```text
(0, 0)
→ (1, 0)
→ (2, 0)
→ (3, 0)
→ (4, 0)
→ (4, 1)
→ (4, 2)
→ (4, 3)
→ (4, 4)