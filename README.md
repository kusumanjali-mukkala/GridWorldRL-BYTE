# Gladiator - Grid World Reinforcement Learning Agent

## Project Overview

This project implements a custom **Grid World Reinforcement Learning environment** and trains an agent using the **Tabular Q-Learning algorithm**.

The agent learns how to move from a starting position to a goal while avoiding walls and traps. Through repeated training episodes, the agent learns which actions provide better long-term rewards.

## Task

**Task 9 - Gladiator: Grid World RL Agent**

## Objective

The main objective is to train a Reinforcement Learning agent that can learn an efficient path from the starting position to the goal.

The agent receives rewards and penalties based on its actions and gradually learns the best action for each state.

---

## Environment

The environment is a custom **5 × 5 Grid World**.

### Grid Layout

```text
S  .  .  #  .
.  #  .  #  .
.  .  X  .  .
.  #  .  X  .
.  .  .  .  G
```

### Symbols

| Symbol | Meaning |
|---|---|
| `S` | Start |
| `G` | Goal |
| `X` | Trap |
| `#` | Wall |
| `.` | Empty cell |

**Starting position:** `(0, 0)`

**Goal position:** `(4, 4)`

---

## Actions

The agent can perform four actions:

- `UP`
- `DOWN`
- `LEFT`
- `RIGHT`

The environment checks the selected action and returns the next state, reward, and whether the episode has finished.

---

## Reward System

| Situation | Reward |
|---|---:|
| Normal movement | `-1` |
| Wall / boundary | `-10` |
| Trap | `-100` |
| Goal | `+100` |

The reward system encourages the agent to reach the goal while avoiding traps, walls, and unnecessary movements.

---

## Algorithm

The project uses **Tabular Q-Learning**, a model-free Reinforcement Learning algorithm.

The agent maintains a **Q-table** containing the estimated value of taking each possible action from each state.

The agent uses an **epsilon-greedy strategy**:

- **Exploration:** try different actions to discover better paths.
- **Exploitation:** choose the action with the highest learned Q-value.

During training, the Q-table is repeatedly updated based on the rewards received by the agent.

---

## Training

The agent was trained for:

**5,000 episodes**

### Training Parameters

| Parameter | Value |
|---|---:|
| Learning rate (`α`) | `0.1` |
| Discount factor (`γ`) | `0.95` |
| Initial epsilon | `1.0` |
| Minimum epsilon | `0.01` |
| Epsilon decay | `0.995` |

---

## Results

The trained agent achieved the following overall training results:

| Metric | Result |
|---|---:|
| Overall Win Rate | **97.38%** |
| Average Reward | **84.06** |
| Average Steps | **8.72** |

---

## Evaluation

After training, the agent successfully reached the goal.

### Learned Path

```text
(0,0)
  ↓
(1,0)
  ↓
(2,0)
  ↓
(3,0)
  ↓
(4,0)
  →
(4,1)
  →
(4,2)
  →
(4,3)
  →
(4,4)
```

### Evaluation Results

- **Steps:** 8
- **Total Reward:** 93
- **Optimal Steps:** 8
- **Path Efficiency:** 100%
- **Status:** SUCCESS

The learned path reaches the goal without entering a trap.

---

## Visualizations

Training generated the following visual outputs:

### Win Rate Curve

`outputs/win_rate_curve.png`

### Reward Curve

`outputs/reward_curve.png`

### Steps Curve

`outputs/steps_curve.png`

### Trained Agent Animation

`outputs/trained_agent.gif`

The GIF shows the trained agent navigating through the Grid World.

---

## Project Structure

```text
GridWorldRL-BYTE/
│
├── agent.py
├── environment.py
├── train.py
├── evaluate.py
├── visualize.py
├── visualize_agent.py
│
├── models/
│   └── q_table.npy
│
├── outputs/
│   ├── win_rate_curve.png
│   ├── reward_curve.png
│   ├── steps_curve.png
│   ├── trained_agent.gif
│   └── training_data.npz
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Technologies Used

- **Python**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Reinforcement Learning**
- **Q-Learning**
- **Git & GitHub**

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/kusumanjali-mukkala/GridWorldRL-BYTE.git
cd GridWorldRL-BYTE
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the agent

```bash
python train.py
```

### 6. Evaluate the trained agent

```bash
python evaluate.py
```

### 7. Generate visualizations

```bash
python visualize.py
python visualize_agent.py
```

---

## Key Learning Outcomes

Through this project, the following concepts were implemented:

- Reinforcement Learning fundamentals
- State and action representation
- Reward and penalty design
- Q-table representation
- Epsilon-greedy exploration
- Q-Learning training
- Agent evaluation
- Training performance visualization
- Saving and loading a trained Q-table
- Git and GitHub project management

---

## Conclusion

The project demonstrates how a Reinforcement Learning agent can learn to navigate a custom Grid World using **Tabular Q-Learning**.

After training, the agent learned an efficient path from the start state `(0,0)` to the goal state `(4,4)` while avoiding traps and walls.

The project also includes training metrics, evaluation results, a trained Q-table, and visualizations for analyzing the agent's learning performance.
