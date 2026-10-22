# VOIDRUNNER

**Deep Reinforcement Learning for Hazardous-Terrain Navigation**

## Overview

VOIDRUNNER is a 2D platformer environment in which a Deep Q-Network (DQN) agent learns to navigate hazardous terrain by interacting with the environment.

The project explores how reinforcement learning can help an agent make movement decisions, avoid hazards, and reach a goal. The problem is inspired by autonomous robot navigation across difficult terrain.

## Objectives

- Build a custom 2D environment using Pygame and Gymnasium.
- Implement a DQN agent using PyTorch.
- Train the agent using experience replay and a target network.
- Generate multiple terrain layouts with progressively increasing difficulty.
- Evaluate performance on previously unseen terrains.
- Measure success rate, cumulative reward, and failure rate.

## Technology Stack

- Python
- PyTorch
- Gymnasium
- Pygame
- NumPy
- Matplotlib

## Project Structure

- `environment/` — Terrain environment and physics.
- `agent/` — DQN model and replay buffer.
- `training/` — Training loop and checkpoint management.
- `evaluation/` — Agent evaluation and performance metrics.
- `checkpoints/` — Saved model checkpoints.
- `results/` — Training logs and evaluation outputs.
- `main.py` — Main entry point.

## Setup

Create and activate the Conda environment:

```bash
conda create -n voidrunner python=3.11
conda activate voidrunner
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the environment:

```bash
python main.py
```

## Development Roadmap

1. Environment physics and validation.
2. Multiple terrain layouts and difficulty levels.
3. DQN implementation.
4. Training and curriculum learning.
5. Evaluation on unseen terrains.
6. Optional energy-aware navigation.


