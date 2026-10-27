# VoidRunner — DQN Robot Navigation

VoidRunner is a small 2D side-view simulation of an autonomous rover crossing hazardous terrain. The agent learns to move towards the goal while dealing with gaps, rocks, different ground surfaces, an energy budget, and a deadline.

## Run the trained model

This repository includes the selected trained checkpoint at:

`models/voidrunner_fullterrain_best.pt`

On Windows with Python 3.11 installed, open PowerShell in the repository folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python demo.py
```

A Pygame window opens and the trained DQN runs one course. The console reports its chosen actions, position, energy use, and whether the episode succeeded. The demo uses the learned DQN greedily (`epsilon = 0`) and does not call the rule-based baseline.

Try a harder course or a repeatable seed:

```powershell
python demo.py --difficulty medium --seed 10000
python demo.py --difficulty hard --seed 10000
```

For a headless run without a graphics window:

```powershell
python demo.py --difficulty hard --seed 10000 --no-render
```


## Project structure

- `environment/terrain_env.py` — Gymnasium environment, movement physics, observations, rewards and terminal conditions.
- `environment/terrain_generator.py` — seeded terrain generation and layout validation.
- `agent/dqn.py` — Deep Q-Network, target network, replay buffer integration and checkpoint loading.
- `agent/replay_buffer.py` — experience replay.
- `training/train_full_guided.py` — transfer-learning and full-terrain fine-tuning pipeline; additional obstacle-guidance rewards are used during training only.
- `evaluation/evaluate.py` — held-out evaluation and rule-based/random comparisons.
- `demo.py` — simple visual entry point for the selected trained model.
- `models/voidrunner_fullterrain_best.pt` — selected trained full-terrain DQN checkpoint.
- `docs/voidrunner_final_report.pdf` and `docs/voidrunner_final_report.docx` — final project report in PDF and editable Word formats.

## Model and learning approach

The policy is a Deep Q-Network (DQN) with 18 numeric observations and six discrete actions:

| Action index | Action |
|---:|---|
| 0 | Brake |
| 1 | Walk forward |
| 2 | Sprint forward |
| 3 | Hop |
| 4 | Leap |
| 5 | Reverse |

The observation includes speed, vertical velocity, grounded state, remaining energy, route progress, surface type, the next two obstacle types and distances, obstacle size, and landing clearance. Training uses experience replay, a target network, epsilon-greedy exploration, curriculum learning, transfer initialization from a single-gap DQN, and training-only reward shaping. During evaluation and in `demo.py`, the additional shaping rewards are not added; success means the environment's goal condition was reached.

## Evaluation results

The selected checkpoint `voidrunner_multigap_transfer_seed46/best.pt` was evaluated on 30 seeded layouts per difficulty, using seed range 10000–10029. The same seed range was used for the DQN, rule-based controller and random controller for matched comparisons.

| Difficulty | DQN | Rule-based baseline | Random baseline |
|---|---:|---:|---:|
| Easy | 30/30 (100%) | 30/30 (100%) | 0/30 (0%) |
| Medium | 27/30 (90%) | 30/30 (100%) | 0/30 (0%) |
| Hard | 24/30 (80%) | 30/30 (100%) | 0/30 (0%) |
| **Total** | **81/90 (90%)** | **90/90 (100%)** | **0/90 (0%)** |

The DQN completed 81 of 90 evaluation episodes. It is not perfect: failures remained on Medium and Hard terrain. The rule-based controller is a comparison baseline, not the submitted RL agent. A later experimental observation adapter that represented rocks as gaps reduced performance on this seed range and was rejected; the checkpoint and results above use native obstacle observations.

Evaluation output can be reproduced from the repository root with:

```powershell
python -m evaluation.evaluate --compare --checkpoint .\models\voidrunner_fullterrain_best.pt --difficulty all --episodes 30 --seed-start 10000 --output-dir .\results\final_evaluation --device auto
```

## Environment validation

Run the environment tests with:

```powershell
python -m main --validate
python -m agent.dqn --unit-test
```

These verify environment contracts and DQN/checkpoint utilities. Training is not required just to run the demo.

## Notes

- The environment is a high-level 2D simulation, not a joint-level or hardware-validated robot model.
- Random seeds make layouts reproducible in the current code version; they do not guarantee performance on every unseen layout.
- The model checkpoint must be kept together with the matching source code and environment version.
