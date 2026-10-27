"""VoidRunner Phase 11: transfer the single-gap DQN to full multi-obstacle terrain.

The single-gap checkpoint initializes shared layers and actions 0-4. Its Reverse
head is deliberately not copied because Reverse was masked during single-gap
training. Full-terrain fine-tuning starts with all six actions available, a new
replay buffer, a fresh optimizer, and unshaped held-out evaluations.

This remains DQN learning: the checkpoint supplies initialization, not actions.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = _SCRIPT_PATH.parents[1] if _SCRIPT_PATH.parent.name == "training" else _SCRIPT_PATH.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRAINING_CHECKPOINT_FORMAT = "voidrunner_training_resume"
TRAINING_CHECKPOINT_VERSION = 2
MODEL_MILESTONE_EPISODES = {1, 2, 11, 15}
MODEL_CHECKPOINT_INTERVAL = 25

DEFAULT_CONFIG: dict[str, Any] = {
    "episodes": 20,
    "max_steps": 1200,
    "seed": 42,
    "hidden_size": 128,
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "batch_size": 64,
    "replay_capacity": 50_000,
    "min_replay_size": 5_000,
    "target_update_frequency": 1_000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 150_000,
    "eval_interval": 50,
    "eval_episodes_per_difficulty": 5,
    "curriculum_check_interval": 50,
    "probe_episodes": 20,
    "curriculum_success_threshold": 0.70,
    "resume_interval": 25,
    "render": False,
    "device": "auto",
    # Training-only event rewards; evaluation remains completely unshaped.
    "jump_launch_reward": 2.5,
    "airborne_obstacle_reward": 0.25,
    "gap_clearance_reward": 7.0,
    "rock_clearance_reward": 5.0,
    "approach_distance": 190.0,
    "runup_sprint_reward": 0.4,
    "missed_launch_penalty": 1.25,
    "early_jump_penalty": 0.25,
    "late_jump_penalty": 0.75,
    "reward_shaping_version": "full_terrain_launch_window_v2",
}

CURRICULUM_STAGES: list[dict[str, Any]] = [
    {
        "name": "easy_foundation",
        "weights": {"easy": 1.0, "medium": 0.0, "hard": 0.0},
        "probe_difficulty": "easy",
    },
    {
        "name": "easy_plus_medium",
        "weights": {"easy": 0.8, "medium": 0.2, "hard": 0.0},
        "probe_difficulty": "medium",
    },
    {
        "name": "mixed_ramp",
        "weights": {"easy": 0.50, "medium": 0.35, "hard": 0.15},
        "probe_difficulty": "hard",
    },
    {
        "name": "mixed_full",
        "weights": {"easy": 0.20, "medium": 0.40, "hard": 0.40},
        "probe_difficulty": None,
    },
]
DIFFICULTIES = ("easy", "medium", "hard")
ACTION_METRIC_NAMES = {
    0: "brake",
    1: "walk",
    2: "sprint",
    3: "hop",
    4: "leap",
    5: "reverse",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False")
    return torch.device(name)


def terrain_seed_for_episode(training_seed: int, episode: int) -> int:
    """A stable training-seed range, separate from held-out validation seeds."""
    return int((training_seed * 1_000_003 + episode * 7_919 + 104_729) % (2**31 - 1))


def validation_seed(training_seed: int, split: str, difficulty: str, index: int) -> int:
    """Fixed, reproducible seeds in ranges separate from generated training seeds."""
    split_offsets = {"validation": 0, "curriculum": 500_000}
    difficulty_offsets = {"easy": 0, "medium": 100_000, "hard": 200_000}
    base = 3_000_000_000 + (int(training_seed) % 10_000) * 1_000
    return int((base + split_offsets[split] + difficulty_offsets[difficulty] + index) % (2**32 - 1))


def weighted_difficulty(rng: random.Random, stage_index: int) -> str:
    stage = CURRICULUM_STAGES[stage_index]
    names = [name for name in DIFFICULTIES if stage["weights"].get(name, 0.0) > 0]
    weights = [float(stage["weights"][name]) for name in names]
    return rng.choices(names, weights=weights, k=1)[0]


def greedy_action(agent: DQNAgent, observation: np.ndarray) -> int:
    """Greedy validation action using the same validity mask as training."""
    return agent.select_action(observation, epsilon=0.0)


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def new_obstacle_shaping_state() -> dict[str, Any]:
    """Per-episode state for bounded event-based obstacle shaping."""
    return {
        "jump_targets": set(),
        "runup_paid": set(),
        "launch_counts": {},
        "crossed_airborne": set(),
        "clearance_paid": set(),
        "missed_window_paid": set(),
    }


def _launch_window(obstacle: dict[str, Any]) -> tuple[float, float]:
    """A broad *reward window*, not a prescribed action or policy.

    Distances are measured from the player's right edge to the obstacle's near
    edge. Wider gaps and taller rocks get a somewhat earlier launch window.
    """
    if obstacle["kind"] == "gap":
        width = float(obstacle["end"] - obstacle["start"])
        return 18.0, min(135.0, 72.0 + 0.70 * width)
    height = float(obstacle["top_delta"])
    return 20.0, min(135.0, 68.0 + 1.15 * height)


def apply_full_terrain_reward_shaping(
    env: Any,
    *,
    action: int,
    x_before: float,
    speed_before: float,
    was_grounded: bool,
    info: dict[str, Any],
    state: dict[str, Any],
    config: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Apply training-only feedback for obstacle approach, launch timing, and clearance.

    This does NOT force an action. Reward is given for a real sprint run-up,
    a physical jump initiated in a geometry-dependent window, safe airborne
    passage, and landing past the obstacle. Missing the launch window while
    remaining grounded gets a one-off penalty. Evaluation uses the base reward
    and success flag only.
    """
    additions = {
        "runup": 0.0,
        "jump_launch": 0.0,
        "bad_jump": 0.0,
        "missed_window": 0.0,
        "airborne": 0.0,
        "clearance": 0.0,
    }
    x_after = float(info.get("x", getattr(env, "x", x_before)))
    y_after = float(info.get("y", getattr(env, "y", 0.0)))
    grounded_after = bool(info.get("grounded", getattr(env, "grounded", True)))
    vy_after = float(info.get("vertical_velocity", getattr(env, "vy", 0.0)))
    speed_after = float(info.get("speed", getattr(env, "speed", speed_before)))
    player_width = float(getattr(env, "PLAYER_WIDTH", 28))
    player_height = float(getattr(env, "PLAYER_HEIGHT", 36))
    ground_y = float(getattr(env, "GROUND_Y", 390))
    approach_distance = float(config.get("approach_distance", 190.0))

    obstacles: list[dict[str, Any]] = []
    for raw_x, raw_width in getattr(env, "gaps", []):
        ox, width = float(raw_x), float(raw_width)
        obstacles.append({
            "key": ("gap", int(raw_x), int(raw_width), 0),
            "kind": "gap", "start": ox, "end": ox + width,
            "top": ground_y, "top_delta": 0.0,
        })
    for raw_x, raw_width, raw_height in getattr(env, "rocks", []):
        ox, width, height = float(raw_x), float(raw_width), float(raw_height)
        obstacles.append({
            "key": ("rock", int(raw_x), int(raw_width), int(raw_height)),
            "kind": "rock", "start": ox, "end": ox + width,
            "top": ground_y - height, "top_delta": height,
        })
    obstacles.sort(key=lambda item: item["start"])
    if not obstacles:
        return 0.0, additions

    def distance_to(obstacle: dict[str, Any], x: float) -> float:
        return float(obstacle["start"] - (x + player_width))

    def window(obstacle: dict[str, Any]) -> tuple[float, float]:
        return _launch_window(obstacle)

    # Reward one actual sprint run-up per obstacle once speed is already high.
    # It encourages acceleration before the launch area, not endless sprinting.
    if int(action) == 2 and was_grounded:
        for obstacle in obstacles:
            key = obstacle["key"]
            if key in state["runup_paid"] or obstacle["end"] < x_before:
                continue
            distance = distance_to(obstacle, x_before)
            _, launch_far = window(obstacle)
            if launch_far < distance <= launch_far + 95.0 and speed_after >= 5.5:
                additions["runup"] += float(config.get("runup_sprint_reward", 0.4))
                state["runup_paid"].add(key)
                break

    # A physical jump is distinguished from merely pressing Hop/Leap while
    # airborne, out of energy, or otherwise unable to take off.
    physically_launched = (
        was_grounded and int(action) in (3, 4) and not grounded_after and vy_after < 0.0
    )
    launched_key = None
    if physically_launched:
        candidates = []
        for obstacle in obstacles:
            if obstacle["end"] < x_before:
                continue
            distance = distance_to(obstacle, x_before)
            if -player_width <= distance <= approach_distance:
                candidates.append((max(0.0, distance), obstacle, distance))
        if candidates:
            _, target, distance = min(candidates, key=lambda item: abs(item[2]))
            key = target["key"]
            launch_near, launch_far = window(target)
            if launch_near <= distance <= launch_far:
                launched_key = key
                state["jump_targets"].add(key)
                counts = state["launch_counts"]
                count = int(counts.get(key, 0))
                if count < 2:
                    base = float(config.get("jump_launch_reward", 2.5))
                    # Higher horizontal speed is rewarded but is not mandatory.
                    additions["jump_launch"] += base if speed_before >= 5.5 else base * 0.6
                    counts[key] = count + 1
            elif distance > launch_far:
                additions["bad_jump"] -= float(config.get("early_jump_penalty", 0.25))
            else:
                additions["bad_jump"] -= float(config.get("late_jump_penalty", 0.75))

    # One-off penalty when the player passes the launch window still grounded.
    # This gives immediate feedback for running straight at a hazard without jumping.
    if was_grounded and grounded_after and not physically_launched:
        for obstacle in obstacles:
            key = obstacle["key"]
            if key in state["jump_targets"] or key in state["missed_window_paid"]:
                continue
            if obstacle["end"] < x_before:
                continue
            before_distance = distance_to(obstacle, x_before)
            after_distance = distance_to(obstacle, x_after)
            launch_near, launch_far = window(obstacle)
            if before_distance > launch_near and after_distance <= launch_near:
                additions["missed_window"] -= float(config.get("missed_launch_penalty", 1.25))
                state["missed_window_paid"].add(key)
                break

    # Dense but bounded signal only while physically airborne and safely above
    # the obstacle. The jump must have been initiated in the preferred window.
    player_bottom = y_after + player_height
    airborne_caps = state.setdefault("airborne_reward_count", {})
    for obstacle in obstacles:
        key = obstacle["key"]
        if key not in state["jump_targets"]:
            continue
        horizontal_overlap = x_after + player_width > obstacle["start"] and x_after < obstacle["end"]
        if not horizontal_overlap or grounded_after:
            continue
        if obstacle["kind"] == "gap":
            safely_clear = player_bottom <= ground_y - 4.0
        else:
            safely_clear = player_bottom <= obstacle["top"] + 2.0
        if safely_clear and int(airborne_caps.get(key, 0)) < 12:
            additions["airborne"] += float(config.get("airborne_obstacle_reward", 0.25))
            airborne_caps[key] = int(airborne_caps.get(key, 0)) + 1
            state["crossed_airborne"].add(key)

    # Event reward only after safe airborne overlap and landing past the far edge.
    if grounded_after:
        for obstacle in obstacles:
            key = obstacle["key"]
            if key in state["clearance_paid"] or key not in state["crossed_airborne"]:
                continue
            if x_after < obstacle["end"]:
                continue
            if obstacle["kind"] == "gap":
                additions["clearance"] += float(config.get("gap_clearance_reward", 7.0))
            else:
                additions["clearance"] += float(config.get("rock_clearance_reward", 5.0))
            state["clearance_paid"].add(key)

    return float(sum(additions.values())), additions


def evaluation_episode(
    TerrainEnv: Any,
    agent: DQNAgent,
    difficulty: str,
    terrain_seed: int,
    max_steps: int,
    snapshot_path: Path | None = None,
) -> dict[str, Any]:
    env = TerrainEnv(difficulty=difficulty, render_mode=None, max_steps=max_steps)
    try:
        observation, reset_info = env.reset(seed=terrain_seed, options={"terrain_seed": terrain_seed})
        if snapshot_path is not None:
            from environment.terrain_generator import save_layout
            save_layout(reset_info["terrain_layout"], snapshot_path)
        total_reward = 0.0
        final_info: dict[str, Any] = {}
        for _ in range(max_steps):
            action = greedy_action(agent, observation)
            observation, reward, terminated, truncated, final_info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return {
            "difficulty": difficulty,
            "terrain_seed": int(terrain_seed),
            "return": float(total_reward),
            "steps": int(final_info.get("steps", getattr(env, "steps", max_steps))),
            "success": bool(final_info.get("success", False)),
            "failure_reason": final_info.get("failure_reason") or ("time_limit" if final_info.get("truncated") else None),
            "distance_reached": float(final_info.get("distance_reached", 0.0)),
            "energy_consumed": float(final_info.get("energy_consumed", 0.0)),
            "energy_remaining": float(final_info.get("energy_remaining", 0.0)),
            "terrain_snapshot": (
                str(snapshot_path.relative_to(PROJECT_ROOT))
                if snapshot_path is not None and snapshot_path.is_relative_to(PROJECT_ROOT)
                else (str(snapshot_path) if snapshot_path is not None else None)
            ),
        }
    finally:
        env.close()


def evaluate_suite(
    TerrainEnv: Any,
    agent: DQNAgent,
    *,
    training_seed: int,
    max_steps: int,
    episodes_per_difficulty: int,
    snapshot_root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    detailed: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for difficulty in DIFFICULTIES:
        records = []
        for index in range(episodes_per_difficulty):
            seed = validation_seed(training_seed, "validation", difficulty, index)
            snapshot_path = (
                snapshot_root / difficulty / f"validation_{difficulty}_seed_{seed}.json"
                if snapshot_root is not None else None
            )
            records.append(evaluation_episode(
                TerrainEnv, agent, difficulty, seed, max_steps, snapshot_path=snapshot_path
            ))
        detailed.extend(records)
        successful = [row for row in records if row["success"]]
        summaries[difficulty] = {
            "episodes": len(records),
            "successes": len(successful),
            "success_rate": len(successful) / max(1, len(records)),
            "mean_return": float(np.mean([row["return"] for row in records])),
            "mean_steps": float(np.mean([row["steps"] for row in records])),
            "mean_energy_consumed": float(np.mean([row["energy_consumed"] for row in records])),
            "mean_energy_per_success": mean_or_none([row["energy_consumed"] for row in successful]),
            "mean_steps_per_success": mean_or_none([float(row["steps"]) for row in successful]),
            "failure_reasons": {
                reason: sum(row["failure_reason"] == reason for row in records)
                for reason in sorted({str(row["failure_reason"]) for row in records if row["failure_reason"]})
            },
        }
    summaries["overall_success_rate"] = float(
        np.mean([summaries[difficulty]["success_rate"] for difficulty in DIFFICULTIES])
    )
    all_successes = [row for row in detailed if row["success"]]
    summaries["overall_mean_energy_per_success"] = mean_or_none(
        [row["energy_consumed"] for row in all_successes]
    )
    summaries["overall_mean_steps_per_success"] = mean_or_none(
        [float(row["steps"]) for row in all_successes]
    )
    return summaries, detailed


def validation_key(summary: dict[str, Any]) -> tuple[float, float, float]:
    """Compare checkpoints lexicographically: success first, energy then steps."""
    success = float(summary["overall_success_rate"])
    energy = summary["overall_mean_energy_per_success"]
    steps = summary["overall_mean_steps_per_success"]
    # If no route was completed, don't let low energy from early failure appear good.
    energy_key = -float(energy) if energy is not None else -1_000_000.0
    steps_key = -float(steps) if steps is not None else -1_000_000.0
    return success, energy_key, steps_key


def append_csv(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    if not needs_header:
        with path.open("r", newline="", encoding="utf-8") as check_handle:
            existing_header = next(csv.reader(check_handle), [])
        if existing_header != fieldnames:
            raise ValueError(f"CSV schema mismatch in {path}; refusing to append incompatible rows")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")
        handle.flush()


def write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_training_resume(
    path: Path,
    *,
    agent: DQNAgent,
    episode: int,
    global_step: int,
    curriculum_stage: int,
    training_rng: random.Random,
    curriculum_rng: random.Random,
    best_key: tuple[float, float, float] | None,
    best_summary: dict[str, Any] | None,
    run_id: str,
    results_dir: Path,
    checkpoints_dir: Path,
    run_config: dict[str, Any],
) -> None:
    """Save complete episode-boundary state to one replaceable resume file."""
    payload = {
        "checkpoint_format": TRAINING_CHECKPOINT_FORMAT,
        "checkpoint_format_version": TRAINING_CHECKPOINT_VERSION,
        "run_id": run_id,
        "episode": int(episode),
        "global_step": int(global_step),
        "curriculum_stage": int(curriculum_stage),
        "environment_version": agent.environment_version,
        "architecture": {
            "state_size": agent.state_size,
            "action_size": agent.action_size,
            "hidden_size": agent.hidden_size,
        },
        "observation_features": list(agent.observation_features),
        "action_names": dict(agent.action_names),
        "agent_config": {
            "learning_rate": agent.learning_rate,
            "gamma": agent.gamma,
            "batch_size": agent.batch_size,
            "min_replay_size": agent.min_replay_size,
            "replay_capacity": agent.replay_buffer.capacity,
            "target_update_frequency": agent.target_update_frequency,
            "seed": agent.seed,
        },
        "online_network_state_dict": agent.online_network.state_dict(),
        "target_network_state_dict": agent.target_network.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "learn_steps": agent.learn_steps,
        "replay_buffer_state": agent.replay_buffer.state_dict(),
        "agent_rng_state": agent.rng.getstate(),
        "training_rng_state": training_rng.getstate(),
        "curriculum_rng_state": curriculum_rng.getstate(),
        "python_global_rng_state": random.getstate(),
        "numpy_global_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "best_validation_key": list(best_key) if best_key is not None else None,
        "best_validation_summary": best_summary,
        "results_dir": str(results_dir.relative_to(PROJECT_ROOT)),
        "checkpoints_dir": str(checkpoints_dir.relative_to(PROJECT_ROOT)),
        "run_config": run_config,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_training_resume(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    # Resume checkpoints are trusted files created by this project. weights_only=False
    # is needed because they contain Python/NumPy RNG and replay-buffer metadata.
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("Invalid resume checkpoint: payload is not a dictionary")
    if payload.get("checkpoint_format") != TRAINING_CHECKPOINT_FORMAT:
        raise ValueError("This is not a full VoidRunner training-resume checkpoint")
    if payload.get("checkpoint_format_version") != TRAINING_CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported resume checkpoint version: {payload.get('checkpoint_format_version')}")
    return payload


def restore_agent_and_rng(
    payload: dict[str, Any],
    *,
    agent: DQNAgent,
    training_rng: random.Random,
    curriculum_rng: random.Random,
) -> None:
    expected_architecture = {
        "state_size": agent.state_size,
        "action_size": agent.action_size,
        "hidden_size": agent.hidden_size,
    }
    if payload.get("architecture") != expected_architecture:
        raise ValueError(
            f"Resume model architecture mismatch: saved={payload.get('architecture')}, current={expected_architecture}"
        )
    if payload.get("environment_version") != agent.environment_version:
        raise ValueError(
            f"Environment version mismatch: resume={payload.get('environment_version')!r}, "
            f"current={agent.environment_version!r}. Start a new run instead of resuming across environment versions."
        )
    if tuple(payload.get("observation_features", ())) != tuple(agent.observation_features):
        raise ValueError("Observation feature order changed; refusing to resume")
    saved_actions = {str(k): str(v) for k, v in payload.get("action_names", {}).items()}
    current_actions = {str(k): str(v) for k, v in agent.action_names.items()}
    if saved_actions != current_actions:
        raise ValueError("Action meanings changed; refusing to resume")
    agent.online_network.load_state_dict(payload["online_network_state_dict"], strict=True)
    agent.target_network.load_state_dict(payload["target_network_state_dict"], strict=True)
    agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
    agent.learn_steps = int(payload.get("learn_steps", 0))
    agent.replay_buffer.load_state_dict(payload["replay_buffer_state"])
    agent.rng.setstate(payload["agent_rng_state"])
    training_rng.setstate(payload["training_rng_state"])
    curriculum_rng.setstate(payload["curriculum_rng_state"])
    random.setstate(payload["python_global_rng_state"])
    np.random.set_state(payload["numpy_global_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_rng_states") is not None:
        try:
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_states"]])
        except Exception as exc:
            raise ValueError("Could not restore CUDA RNG state on this device") from exc
    agent.target_network.eval()


def make_run_directories(run_id: str) -> tuple[Path, Path]:
    results_dir = PROJECT_ROOT / "results" / run_id
    checkpoints_dir = PROJECT_ROOT / "checkpoints" / run_id
    results_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir.mkdir(parents=True, exist_ok=False)
    (results_dir / "terrain_snapshots").mkdir(parents=True, exist_ok=True)
    return results_dir, checkpoints_dir


def save_model_snapshot(
    agent: DQNAgent,
    checkpoints_dir: Path,
    episode: int,
    global_step: int,
    metadata: dict[str, Any],
) -> Path:
    path = checkpoints_dir / f"episode_{episode:06d}_model.pt"
    agent.save_checkpoint(path, episode=episode, global_step=global_step, extra_metadata=metadata)
    return path


def write_evaluation_rows(
    results_dir: Path,
    *,
    run_id: str,
    episode: int,
    eval_type: str,
    summary: dict[str, Any],
    detailed: list[dict[str, Any]],
) -> None:
    path = results_dir / "evaluation_log.csv"
    fields = [
        "run_id", "training_episode", "eval_type", "difficulty", "episodes", "successes",
        "success_rate", "mean_return", "mean_steps", "mean_energy_consumed",
        "mean_energy_per_success", "mean_steps_per_success", "failure_reasons", "terrain_seeds",
    ]
    for difficulty in DIFFICULTIES:
        row = summary[difficulty]
        append_csv(path, fields, {
            "run_id": run_id,
            "training_episode": episode,
            "eval_type": eval_type,
            "difficulty": difficulty,
            "episodes": row["episodes"],
            "successes": row["successes"],
            "success_rate": row["success_rate"],
            "mean_return": row["mean_return"],
            "mean_steps": row["mean_steps"],
            "mean_energy_consumed": row["mean_energy_consumed"],
            "mean_energy_per_success": row["mean_energy_per_success"],
            "mean_steps_per_success": row["mean_steps_per_success"],
            "failure_reasons": json.dumps(row["failure_reasons"], sort_keys=True),
            "terrain_seeds": json.dumps([d["terrain_seed"] for d in detailed if d["difficulty"] == difficulty]),
        })
    append_csv(path, fields, {
        "run_id": run_id,
        "training_episode": episode,
        "eval_type": eval_type,
        "difficulty": "OVERALL",
        "episodes": sum(summary[d]["episodes"] for d in DIFFICULTIES),
        "successes": sum(summary[d]["successes"] for d in DIFFICULTIES),
        "success_rate": summary["overall_success_rate"],
        "mean_return": float(np.mean([row["return"] for row in detailed])) if detailed else None,
        "mean_steps": float(np.mean([row["steps"] for row in detailed])) if detailed else None,
        "mean_energy_consumed": float(np.mean([row["energy_consumed"] for row in detailed])) if detailed else None,
        "mean_energy_per_success": summary["overall_mean_energy_per_success"],
        "mean_steps_per_success": summary["overall_mean_steps_per_success"],
        "failure_reasons": "{}",
        "terrain_seeds": json.dumps([d["terrain_seed"] for d in detailed]),
    })
    append_jsonl(results_dir / "evaluation_detail.jsonl", {
        "training_episode": episode,
        "eval_type": eval_type,
        "summary": summary,
        "episodes": detailed,
    })


def transfer_single_gap_weights(agent: Any, checkpoint_path: Path, env_module: Any) -> dict[str, Any]:
    """Transfer trained single-gap Q weights into a fresh full-terrain DQN.

    Copies all shared hidden layers and the five action heads trained in the
    single-gap pilot (brake, walk, sprint, hop, leap). The source pilot masked
    Reverse during both action selection and TD bootstrapping, so its Reverse
    output head was not meaningfully trained. The target Reverse head is set to
    a neutral zero-output initialization and is learned normally in the full
    environment. Replay, optimizer state, epsilon schedule, counters, and action
    mask are never copied.
    """
    source = Path(checkpoint_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Transfer checkpoint does not exist: {source}")
    try:
        try:
            payload = torch.load(source, map_location=agent.device, weights_only=True)
        except TypeError:
            payload = torch.load(source, map_location=agent.device)
    except Exception as exc:
        raise ValueError(f"Could not safely load transfer checkpoint: {source}") from exc

    if not isinstance(payload, dict) or payload.get("checkpoint_format") != "voidrunner_dqn":
        raise ValueError("Transfer source must be a versioned VoidRunner DQN model checkpoint (.pt), not a resume file.")
    architecture = payload.get("architecture", {})
    expected = {
        "state_size": int(agent.state_size),
        "action_size": int(agent.action_size),
        "hidden_size": int(agent.hidden_size),
    }
    if architecture != expected:
        raise ValueError(f"Transfer architecture mismatch: source={architecture}; target={expected}.")
    if expected != {"state_size": 18, "action_size": 6, "hidden_size": 128}:
        raise ValueError("This transfer path is configured for the 18-observation / 6-action / 128-hidden-unit VoidRunner model.")

    source_features = tuple(payload.get("observation_features", ()))
    target_features = tuple(getattr(env_module, "OBSERVATION_FEATURES", ()))
    if source_features and target_features and source_features != target_features:
        raise ValueError("Transfer checkpoint observation feature order differs from the current environment.")
    source_actions = {str(k): str(v) for k, v in payload.get("action_names", {}).items()}
    target_actions = {str(k): str(v) for k, v in getattr(env_module, "ACTION_NAMES", {}).items()}
    if source_actions and target_actions and source_actions != target_actions:
        raise ValueError(f"Transfer action meanings differ: source={source_actions}; target={target_actions}.")

    source_mask = payload.get("permanently_masked_actions")
    if source_mask is None:
        source_mask = payload.get("agent_config", {}).get("permanently_masked_actions", [])
    source_mask = tuple(sorted(int(x) for x in source_mask))
    if source_mask != (5,):
        raise ValueError(
            f"Expected the successful single-gap checkpoint with Reverse (action 5) masked; source mask={source_mask}."
        )
    target_mask = tuple(sorted(int(x) for x in getattr(agent, "permanently_masked_actions", ())))
    if target_mask:
        raise ValueError(f"Full-terrain target must expose all actions; its permanent action mask is {target_mask}.")

    source_state = payload.get("online_network_state_dict")
    if not isinstance(source_state, dict):
        raise ValueError("Transfer checkpoint has no online_network_state_dict.")
    target_state = {key: value.detach().clone() for key, value in agent.online_network.state_dict().items()}

    # The shared feature extractor learned from successful single-gap crossings.
    shared_keys = ("network.0.weight", "network.0.bias", "network.2.weight", "network.2.bias")
    for key in shared_keys:
        if key not in source_state or key not in target_state or source_state[key].shape != target_state[key].shape:
            raise ValueError(f"Missing or incompatible shared layer tensor: {key}")
        target_state[key] = source_state[key].detach().clone().to(target_state[key].device)

    # Transfer the learned Q outputs for actions that were genuinely available.
    for key in ("network.4.weight", "network.4.bias"):
        if key not in source_state or key not in target_state or source_state[key].shape != target_state[key].shape:
            raise ValueError(f"Missing or incompatible action-head tensor: {key}")
        target_state[key][:5] = source_state[key][:5].detach().to(target_state[key].device)
        # Reverse had no learning signal in the one-way pilot. Start it at a
        # neutral Q estimate instead of copying an untrained random source head.
        target_state[key][5].zero_()

    agent.online_network.load_state_dict(target_state, strict=True)
    agent.target_network.load_state_dict(agent.online_network.state_dict(), strict=True)
    agent.target_network.eval()
    agent.learn_steps = 0
    # New optimizer is deliberate: this is fine-tuning from weights, not a resume.
    agent.optimizer = torch.optim.Adam(agent.online_network.parameters(), lr=agent.learning_rate)

    source_metadata = payload.get("extra_metadata", {})
    return {
        "method": "weights_only_transfer",
        "source_checkpoint": str(source),
        "source_episode": int(payload.get("episode", 0)),
        "source_global_step": int(payload.get("global_step", 0)),
        "source_masked_actions": list(source_mask),
        "target_masked_actions": [],
        "transferred_hidden_layers": list(shared_keys),
        "transferred_action_heads": ["brake", "walk", "sprint", "hop", "leap"],
        "reverse_head_initialization": "zero_output_neutral_initialization_then_full_terrain_learning",
        "replay_optimizer_and_counters": "fresh",
        "source_experiment": source_metadata.get("experiment", "unknown"),
        "target_evaluation_uses_training_shaping": False,
    }


def run_training(args: argparse.Namespace) -> None:
    from agent.dqn import DQNAgent

    # Import Gymnasium environment only after argument parsing, so --help works
    # even in an environment where Gymnasium has not yet been installed.
    from environment import terrain_env as env_module
    from environment.terrain_env import TerrainEnv
    from environment.terrain_generator import GENERATOR_VERSION, save_layout

    resume_payload: dict[str, Any] | None = None
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("Use --init-checkpoint only for a new fine-tuning run; it cannot be combined with --resume.")
    if args.resume is not None:
        # Device is selected before loading the resume file; its tensors are mapped
        # to the actual current device.
        seed_hint = args.seed if args.seed is not None else int(DEFAULT_CONFIG["seed"])
        device = choose_device(args.device or "auto")
        resume_payload = load_training_resume(args.resume, device)
        run_config = dict(resume_payload["run_config"])
        if run_config.get("reward_shaping_version") != "full_terrain_launch_window_v2":
            raise ValueError(
                "This training script uses a different reward target. Start a fresh Phase 10 run; "
                "do not resume a Phase 7B or Phase 8 checkpoint."
            )
        if args.episodes is not None:
            run_config["episodes"] = int(args.episodes)
        run_config["device"] = str(device)
        if args.run_id is not None or args.render:
            raise ValueError("When resuming, do not supply --run-id or --render; use the saved run configuration.")
        if any(getattr(args, name, None) is not None for name in (
            "seed", "max_steps", "batch_size", "min_replay_size", "replay_capacity",
            "learning_rate", "gamma", "hidden_size", "target_update_frequency",
            "epsilon_start", "epsilon_end", "epsilon_decay_steps", "eval_interval",
            "eval_episodes_per_difficulty", "curriculum_check_interval", "probe_episodes",
            "curriculum_success_threshold", "resume_interval",
            "jump_launch_reward", "airborne_obstacle_reward", "gap_clearance_reward",
            "rock_clearance_reward", "approach_distance", "runup_sprint_reward",
            "missed_launch_penalty", "early_jump_penalty", "late_jump_penalty",
        )):
            raise ValueError("When resuming, only --episodes and --device may override the saved run configuration.")
        run_id = str(resume_payload["run_id"])
        results_dir = PROJECT_ROOT / str(resume_payload["results_dir"])
        checkpoints_dir = PROJECT_ROOT / str(resume_payload["checkpoints_dir"])
        if not results_dir.is_dir() or not checkpoints_dir.is_dir():
            raise FileNotFoundError("The results/checkpoints directories referenced by the resume file do not exist")
        # Old resume config carries the original seed; don't use a new seed hint.
        seed = int(run_config["seed"])
    else:
        run_config = dict(DEFAULT_CONFIG)
        run_config["pilot_mode"] = bool(args.pilot)
        run_config["episodes"] = int(args.episodes if args.episodes is not None else DEFAULT_CONFIG["episodes"])
        if args.seed is not None:
            run_config["seed"] = int(args.seed)
        if args.max_steps is not None:
            run_config["max_steps"] = int(args.max_steps)
        if args.device is not None:
            run_config["device"] = args.device
        seed = int(run_config["seed"])
        run_id = args.run_id or f"energy_v1_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        if Path(run_id).name != run_id or run_id in {"", ".", ".."}:
            raise ValueError("--run-id must be a simple directory name, not a path")
        results_dir, checkpoints_dir = make_run_directories(run_id)
        run_config["initialization_mode"] = "transfer_from_single_gap" if args.init_checkpoint is not None else "random_initialization"
        run_config["init_checkpoint"] = str(args.init_checkpoint.expanduser().resolve()) if args.init_checkpoint is not None else None

        # Optional config overrides are applied only when creating a new run.
        for arg_name, config_name in (
            ("hidden_size", "hidden_size"),
            ("batch_size", "batch_size"),
            ("min_replay_size", "min_replay_size"),
            ("replay_capacity", "replay_capacity"),
            ("learning_rate", "learning_rate"),
            ("gamma", "gamma"),
            ("target_update_frequency", "target_update_frequency"),
            ("epsilon_start", "epsilon_start"),
            ("epsilon_end", "epsilon_end"),
            ("epsilon_decay_steps", "epsilon_decay_steps"),
            ("eval_interval", "eval_interval"),
            ("eval_episodes_per_difficulty", "eval_episodes_per_difficulty"),
            ("curriculum_check_interval", "curriculum_check_interval"),
            ("probe_episodes", "probe_episodes"),
            ("curriculum_success_threshold", "curriculum_success_threshold"),
            ("resume_interval", "resume_interval"),
            ("jump_launch_reward", "jump_launch_reward"),
            ("airborne_obstacle_reward", "airborne_obstacle_reward"),
            ("gap_clearance_reward", "gap_clearance_reward"),
            ("rock_clearance_reward", "rock_clearance_reward"),
            ("approach_distance", "approach_distance"),
            ("runup_sprint_reward", "runup_sprint_reward"),
            ("missed_launch_penalty", "missed_launch_penalty"),
            ("early_jump_penalty", "early_jump_penalty"),
            ("late_jump_penalty", "late_jump_penalty"),
        ):
            value = getattr(args, arg_name, None)
            if value is not None:
                run_config[config_name] = value
        if args.render:
            run_config["render"] = True
        if args.pilot:
            run_config["episodes"] = int(args.episodes if args.episodes is not None else 5)
            run_config["pilot_mode"] = True
            run_config["max_steps"] = min(int(run_config["max_steps"]), 600)
            run_config["batch_size"] = 16
            run_config["replay_capacity"] = 5_000
            run_config["min_replay_size"] = 32
            run_config["target_update_frequency"] = 100
            run_config["eval_interval"] = int(run_config["episodes"])
            run_config["eval_episodes_per_difficulty"] = 2
            run_config["curriculum_check_interval"] = int(run_config["episodes"])
            run_config["probe_episodes"] = 4
            run_config["resume_interval"] = 2
            run_config["render"] = bool(args.render)

    if args.resume is not None and args.pilot:
        raise ValueError("Do not use --pilot with --resume; resume a normal experiment with its saved configuration")
    episodes = int(run_config["episodes"])
    max_steps = int(run_config["max_steps"])
    seed = int(run_config["seed"])
    device = choose_device(str(run_config.get("device", "auto")))
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive")
    if int(run_config["batch_size"]) <= 0:
        raise ValueError("batch_size must be positive")
    if int(run_config["min_replay_size"]) < int(run_config["batch_size"]):
        raise ValueError("min_replay_size must be >= batch_size")
    if int(run_config["replay_capacity"]) < int(run_config["min_replay_size"]):
        raise ValueError("replay_capacity must be >= min_replay_size")
    for name in ("jump_launch_reward", "airborne_obstacle_reward", "gap_clearance_reward", "rock_clearance_reward"):
        if float(run_config.get(name, 0.0)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if float(run_config.get("approach_distance", 160.0)) <= 0.0:
        raise ValueError("approach_distance must be positive")

    seed_everything(seed)
    training_rng = random.Random(seed + 17)
    curriculum_rng = random.Random(seed + 23)
    start_episode = 1
    global_step = 0
    curriculum_stage = 0
    best_key: tuple[float, float, float] | None = None
    best_summary: dict[str, Any] | None = None

    env = TerrainEnv(
        difficulty="easy",
        render_mode="human" if bool(run_config.get("render", False)) else None,
        max_steps=max_steps,
    )
    try:
        if tuple(env.observation_space.shape or ()) != (18,):
            raise ValueError(
                f"Phase 6 expects an 18-element observation, got {env.observation_space.shape}. "
                "Check that the current Phase 2/3 environment is installed."
            )
        if int(env.action_space.n) != 6:
            raise ValueError(
                f"Phase 6 expects six actions, got {env.action_space.n}. "
                "Check that the current environment with Reverse action is installed."
            )
        expected_actions = {str(k): v for k, v in ACTION_METRIC_NAMES.items()}
        actual_actions = {str(k): str(v) for k, v in getattr(env_module, "ACTION_NAMES", {}).items()}
        if actual_actions != expected_actions:
            raise ValueError(
                f"Environment action meanings do not match Phase 6 contract: {actual_actions}"
            )
        agent = DQNAgent.from_env(
            env,
            hidden_size=int(run_config.get("hidden_size", 128)),
            learning_rate=float(run_config["learning_rate"]),
            gamma=float(run_config["gamma"]),
            batch_size=int(run_config["batch_size"]),
            replay_capacity=int(run_config["replay_capacity"]),
            min_replay_size=int(run_config["min_replay_size"]),
            target_update_frequency=int(run_config.get("target_update_frequency", 1_000)),
            device=device,
            seed=seed,
        )
        transfer_metadata: dict[str, Any] | None = None
        if args.init_checkpoint is not None:
            transfer_metadata = transfer_single_gap_weights(agent, args.init_checkpoint, env_module)
            run_config["transfer_initialization"] = transfer_metadata
            print("Transferred policy weights from single-gap model.")
            print("Shared hidden layers + actions brake/walk/sprint/hop/leap copied; Reverse is freshly initialized.")
            print("All six actions are available for full-terrain fine-tuning; replay and optimizer are fresh.")

        if resume_payload is not None:
            restore_agent_and_rng(
                resume_payload,
                agent=agent,
                training_rng=training_rng,
                curriculum_rng=curriculum_rng,
            )
            start_episode = int(resume_payload["episode"]) + 1
            global_step = int(resume_payload["global_step"])
            curriculum_stage = int(resume_payload["curriculum_stage"])
            stored_best_key = resume_payload.get("best_validation_key")
            best_key = tuple(float(x) for x in stored_best_key) if stored_best_key is not None else None
            best_summary = resume_payload.get("best_validation_summary")
            print(f"Resumed run: {run_id} at episode {start_episode - 1}, global step {global_step}")

        # Environment contract and exact reward values are stored with the run.
        config_path = results_dir / "config.json"
        if resume_payload is not None and int(run_config.get("generator_version", GENERATOR_VERSION)) != int(GENERATOR_VERSION):
            raise ValueError(
                f"Terrain generator version mismatch: saved={run_config.get('generator_version')}, current={GENERATOR_VERSION}. Start a new run instead of resuming."
            )
        run_config["generator_version"] = int(GENERATOR_VERSION)
        if not config_path.exists() or resume_payload is None:
            config_payload = {
                "run_id": run_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "environment_version": getattr(env_module, "ENVIRONMENT_VERSION", None),
                "generator_version": GENERATOR_VERSION,
                "observation_features": list(getattr(env_module, "OBSERVATION_FEATURES", ())),
                "action_names": {str(k): v for k, v in getattr(env_module, "ACTION_NAMES", {}).items()},
                "reward_configuration": {
                    name: getattr(TerrainEnv, name)
                    for name in (
                        "PROGRESS_REWARD_WEIGHT", "ENERGY_PENALTY_WEIGHT", "STEP_PENALTY",
                        "HAZARD_CONTACT_PENALTY", "GOAL_REWARD", "FAILURE_PENALTY",
                    )
                },
                "run_config": run_config,
                "training_only_obstacle_reward_shaping": {
                    "jump_launch": run_config.get("jump_launch_reward", 2.5),
                    "airborne_obstacle_per_step": run_config.get("airborne_obstacle_reward", 0.25),
                    "gap_clearance": run_config.get("gap_clearance_reward", 7.0),
                    "rock_clearance": run_config.get("rock_clearance_reward", 5.0),
                    "approach_distance_px": run_config.get("approach_distance", 190.0),
                    "runup_sprint_reward": run_config.get("runup_sprint_reward", 0.4),
                    "missed_launch_penalty": run_config.get("missed_launch_penalty", 1.25),
                    "early_jump_penalty": run_config.get("early_jump_penalty", 0.25),
                    "late_jump_penalty": run_config.get("late_jump_penalty", 0.75),
                    "evaluation_uses_shaping": False,
                },
                "curriculum_stages": CURRICULUM_STAGES,
                "held_out_seed_policy": "Deterministic seed ranges separate from training terrain seeds.",
            }
            write_json_atomic(config_path, config_payload)

        # Record the transferred policy before fine-tuning. This is a true
        # full-terrain evaluation with all six actions enabled and no shaping.
        if transfer_metadata is not None:
            initial_eval_count = min(10, int(run_config["eval_episodes_per_difficulty"]))
            print(f"Evaluating transferred policy before learning ({initial_eval_count} layouts per difficulty)...")
            initial_summary, initial_details = evaluate_suite(
                TerrainEnv,
                agent,
                training_seed=seed,
                max_steps=max_steps,
                episodes_per_difficulty=initial_eval_count,
                snapshot_root=results_dir / "terrain_snapshots" / "transfer_initial",
            )
            write_evaluation_rows(
                results_dir,
                run_id=run_id,
                episode=0,
                eval_type="transfer_initial",
                summary=initial_summary,
                detailed=initial_details,
            )
            print("Initial transferred-policy success:", ", ".join(
                f"{difficulty}={initial_summary[difficulty]['successes']}/{initial_summary[difficulty]['episodes']}"
                for difficulty in DIFFICULTIES
            ))

        if start_episode > episodes:
            print(f"Run already reached episode {start_episode - 1}; requested total is {episodes}.")
            print("Increase --episodes to a larger total to continue.")
            return

        training_fields = [
            "run_id", "episode", "global_step", "curriculum_stage", "curriculum_stage_name",
            "difficulty", "training_seed", "terrain_seed", "terrain_snapshot", "trajectory_trace", "environment_version",
            "episode_return", "learning_return", "reward_shape_runup", "reward_shape_jump_launch",
            "reward_shape_bad_jump", "reward_shape_missed_window", "reward_shape_airborne", "reward_shape_clearance",
            "steps", "success", "failure_reason", "distance_reached",
            "progress_fraction", "energy_consumed", "energy_remaining", "epsilon", "mean_loss",
            "mean_q", "mean_target_q", "reward_progress", "reward_energy_penalty", "reward_step_penalty",
            "reward_hazard_penalty", "reward_terminal", "simulated_time_seconds",
        ] + [f"action_{name}" for name in ACTION_METRIC_NAMES.values()]
        training_log = results_dir / "training_log.csv"

        for episode in range(start_episode, episodes + 1):
            stage_info = CURRICULUM_STAGES[curriculum_stage]
            difficulty = weighted_difficulty(curriculum_rng, curriculum_stage)
            env.difficulty = difficulty
            terrain_seed = terrain_seed_for_episode(seed, episode)
            observation, reset_info = env.reset(
                seed=terrain_seed,
                options={"terrain_seed": terrain_seed},
            )
            layout = reset_info.get("terrain_layout", env.get_layout_snapshot())
            snapshot_path = results_dir / "terrain_snapshots" / f"episode_{episode:06d}_{difficulty}_seed_{terrain_seed}.json"
            save_layout(layout, snapshot_path)
            snapshot_relative = str(snapshot_path.relative_to(PROJECT_ROOT))
            capture_trace = episode in MODEL_MILESTONE_EPISODES or episode % MODEL_CHECKPOINT_INTERVAL == 0
            trace_records: list[dict[str, Any]] = []
            append_jsonl(results_dir / "terrain_manifest.jsonl", {
                "run_id": run_id,
                "episode": episode,
                "difficulty": difficulty,
                "training_seed": seed,
                "terrain_seed": terrain_seed,
                "environment_version": getattr(env_module, "ENVIRONMENT_VERSION", None),
                "generator_version": GENERATOR_VERSION,
                "snapshot_path": snapshot_relative,
            })

            episode_return = 0.0  # Original environment return, without shaping.
            learning_return = 0.0  # Return actually fed to replay, including shaping.
            shaping_state = new_obstacle_shaping_state()
            shaping_sums = {"runup": 0.0, "jump_launch": 0.0, "bad_jump": 0.0, "missed_window": 0.0, "airborne": 0.0, "clearance": 0.0}
            action_counts = {int(index): 0 for index in ACTION_METRIC_NAMES}
            reward_component_sums = {
                "progress": 0.0,
                "energy_penalty": 0.0,
                "step_penalty": 0.0,
                "hazard_penalty": 0.0,
                "terminal": 0.0,
            }
            losses: list[float] = []
            mean_q_values: list[float] = []
            mean_target_values: list[float] = []
            epsilon = float(run_config["epsilon_start"])
            final_info: dict[str, Any] = dict(reset_info)

            for _step in range(max_steps):
                epsilon = max(
                    float(run_config["epsilon_end"]),
                    float(run_config["epsilon_start"])
                    - (float(run_config["epsilon_start"]) - float(run_config["epsilon_end"]))
                    * global_step / float(run_config["epsilon_decay_steps"]),
                )
                action = agent.select_action(observation, epsilon=epsilon)
                x_before = float(env.x)
                speed_before = float(getattr(env, "speed", 0.0))
                was_grounded = bool(env.grounded)
                next_observation, reward, terminated, truncated, info = env.step(action)
                environment_reward = float(reward)
                shaping_reward, shaping_components = apply_full_terrain_reward_shaping(
                    env,
                    action=action,
                    x_before=x_before,
                    speed_before=speed_before,
                    was_grounded=was_grounded,
                    info=info,
                    state=shaping_state,
                    config=run_config,
                )
                learning_reward = environment_reward + shaping_reward
                agent.remember(
                    observation,
                    action,
                    learning_reward,
                    next_observation,
                    terminated=terminated,
                    truncated=truncated,
                )
                update = agent.learn()
                if update is not None:
                    losses.append(update["loss"])
                    mean_q_values.append(update["mean_q"])
                    mean_target_values.append(update["mean_target_q"])

                if capture_trace:
                    trace_records.append({
                        "run_id": run_id,
                        "episode": episode,
                        "step": int(info.get("steps", len(trace_records) + 1)),
                        "global_step": global_step + 1,
                        "observation": np.asarray(observation, dtype=np.float32).tolist(),
                        "action": int(action),
                        "action_name": ACTION_METRIC_NAMES.get(int(action), str(action)),
                        "reward": environment_reward,
                        "learning_reward": learning_reward,
                        "reward_shaping": shaping_components,
                        "next_observation": np.asarray(next_observation, dtype=np.float32).tolist(),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "x": info.get("x"),
                        "y": info.get("y"),
                        "speed": info.get("speed"),
                        "energy_remaining": info.get("energy_remaining"),
                        "reward_components": info.get("reward_components", {}),
                    })

                action_counts[action] += 1
                episode_return += environment_reward
                learning_return += learning_reward
                for name, value in shaping_components.items():
                    shaping_sums[name] += float(value)
                components = info.get("reward_components", {})
                for name in reward_component_sums:
                    reward_component_sums[name] += float(components.get(name, 0.0))
                global_step += 1
                observation = next_observation
                final_info = info
                if bool(run_config.get("render", False)):
                    env.render()
                if terminated or truncated:
                    break

            trace_relative = ""
            if capture_trace:
                trace_path = results_dir / "trajectory_traces" / f"episode_{episode:06d}_trace.jsonl"
                write_jsonl_records(trace_path, trace_records)
                trace_relative = str(trace_path.relative_to(PROJECT_ROOT))

            row: dict[str, Any] = {
                "run_id": run_id,
                "episode": episode,
                "global_step": global_step,
                "curriculum_stage": curriculum_stage,
                "curriculum_stage_name": stage_info["name"],
                "difficulty": difficulty,
                "training_seed": seed,
                "terrain_seed": terrain_seed,
                "terrain_snapshot": snapshot_relative,
                "trajectory_trace": trace_relative,
                "environment_version": getattr(env_module, "ENVIRONMENT_VERSION", None),
                "episode_return": episode_return,
                "learning_return": learning_return,
                "reward_shape_runup": shaping_sums["runup"],
                "reward_shape_jump_launch": shaping_sums["jump_launch"],
                "reward_shape_bad_jump": shaping_sums["bad_jump"],
                "reward_shape_missed_window": shaping_sums["missed_window"],
                "reward_shape_airborne": shaping_sums["airborne"],
                "reward_shape_clearance": shaping_sums["clearance"],
                "steps": int(final_info.get("steps", max_steps)),
                "success": bool(final_info.get("success", False)),
                "failure_reason": final_info.get("failure_reason") or ("time_limit" if final_info.get("truncated") else ""),
                "distance_reached": float(final_info.get("distance_reached", 0.0)),
                "progress_fraction": float(final_info.get("progress_fraction", 0.0)),
                "energy_consumed": float(final_info.get("energy_consumed", 0.0)),
                "energy_remaining": float(final_info.get("energy_remaining", 0.0)),
                "epsilon": epsilon,
                "mean_loss": mean_or_none(losses),
                "mean_q": mean_or_none(mean_q_values),
                "mean_target_q": mean_or_none(mean_target_values),
                "reward_progress": reward_component_sums["progress"],
                "reward_energy_penalty": reward_component_sums["energy_penalty"],
                "reward_step_penalty": reward_component_sums["step_penalty"],
                "reward_hazard_penalty": reward_component_sums["hazard_penalty"],
                "reward_terminal": reward_component_sums["terminal"],
                "simulated_time_seconds": int(final_info.get("steps", max_steps)) / 60.0,
            }
            for action_index, action_name in ACTION_METRIC_NAMES.items():
                row[f"action_{action_name}"] = action_counts[action_index]
            append_csv(training_log, training_fields, row)

            recent_loss = f"{np.mean(losses):.4f}" if losses else "n/a"
            print(
                f"Episode {episode:04d}/{episodes} | stage={curriculum_stage}:{stage_info['name']} "
                f"| {difficulty:6s} | env_return={episode_return:7.3f} "
                f"| learn_return={learning_return:7.3f} | steps={row['steps']:4d} "
                f"| success={row['success']} | energy={row['energy_consumed']:6.2f} "
                f"| eps={epsilon:.3f} | loss={recent_loss}"
            )

            # Historical model-only checkpoints are immutable. Full-resume state
            # is separate and replaceable so it does not flood disk usage.
            if episode in MODEL_MILESTONE_EPISODES or episode % MODEL_CHECKPOINT_INTERVAL == 0:
                saved_model = save_model_snapshot(
                    agent,
                    checkpoints_dir,
                    episode,
                    global_step,
                    {
                        "run_id": run_id,
                        "curriculum_stage": curriculum_stage,
                        "difficulty": difficulty,
                        "terrain_seed": terrain_seed,
                        "terrain_snapshot": snapshot_relative,
                        "environment_version": getattr(env_module, "ENVIRONMENT_VERSION", None),
                        "reward_shaping_version": run_config.get("reward_shaping_version"),
                        "training_only_obstacle_reward_shaping": {
                            name: run_config.get(name) for name in (
                                "jump_launch_reward", "airborne_obstacle_reward",
                                "gap_clearance_reward", "rock_clearance_reward", "approach_distance",
                                "runup_sprint_reward", "missed_launch_penalty",
                                "early_jump_penalty", "late_jump_penalty",
                            )
                        },
                        "reward_configuration": {
                            name: getattr(TerrainEnv, name)
                            for name in (
                                "PROGRESS_REWARD_WEIGHT", "ENERGY_PENALTY_WEIGHT", "STEP_PENALTY",
                                "HAZARD_CONTACT_PENALTY", "GOAL_REWARD", "FAILURE_PENALTY",
                            )
                        },
                    },
                )
                print(f"  Model checkpoint: {saved_model.relative_to(PROJECT_ROOT)}")

            do_evaluate = episode % int(run_config["eval_interval"]) == 0 or episode == episodes
            if do_evaluate:
                summary, detail = evaluate_suite(
                    TerrainEnv,
                    agent,
                    training_seed=seed,
                    max_steps=max_steps,
                    episodes_per_difficulty=int(run_config["eval_episodes_per_difficulty"]),
                    snapshot_root=results_dir / "terrain_snapshots" / "held_out_validation",
                )
                write_evaluation_rows(
                    results_dir,
                    run_id=run_id,
                    episode=episode,
                    eval_type="held_out_validation",
                    summary=summary,
                    detailed=detail,
                )
                print(
                    "  Validation success rates: "
                    + ", ".join(f"{d}={summary[d]['success_rate']:.0%}" for d in DIFFICULTIES)
                    + f" | overall={summary['overall_success_rate']:.1%}"
                )
                candidate_key = validation_key(summary)
                if best_key is None or candidate_key > best_key:
                    best_key = candidate_key
                    best_summary = summary
                    best_path = checkpoints_dir / "best.pt"
                    agent.save_checkpoint(
                        best_path,
                        episode=episode,
                        global_step=global_step,
                        extra_metadata={
                            "run_id": run_id,
                            "selection_metric": "overall held-out success rate, then lower energy/success, then fewer steps/success",
                            "validation_summary": summary,
                            "environment_version": getattr(env_module, "ENVIRONMENT_VERSION", None),
                            "reward_shaping_version": run_config.get("reward_shaping_version"),
                            "training_only_obstacle_reward_shaping": {
                                name: run_config.get(name) for name in (
                                    "jump_launch_reward", "airborne_obstacle_reward",
                                    "gap_clearance_reward", "rock_clearance_reward", "approach_distance",
                                    "runup_sprint_reward", "missed_launch_penalty",
                                    "early_jump_penalty", "late_jump_penalty",
                                )
                            },
                        },
                    )
                    print(f"  New best validation checkpoint: {best_path.relative_to(PROJECT_ROOT)}")

            do_probe = (
                episode % int(run_config["curriculum_check_interval"]) == 0
                or (bool(run_config.get("pilot_mode", False)) and episode == episodes)
            )
            if do_probe and curriculum_stage < len(CURRICULUM_STAGES) - 1:
                probe_difficulty = CURRICULUM_STAGES[curriculum_stage]["probe_difficulty"]
                probe_count = int(run_config["probe_episodes"])
                probe_rows = [
                    evaluation_episode(
                        TerrainEnv,
                        agent,
                        probe_difficulty,
                        validation_seed(seed, "curriculum", probe_difficulty, index),
                        max_steps,
                        snapshot_path=(
                            results_dir / "terrain_snapshots" / "curriculum_probes" / probe_difficulty
                            / f"curriculum_{probe_difficulty}_seed_{validation_seed(seed, 'curriculum', probe_difficulty, index)}.json"
                        ),
                    )
                    for index in range(probe_count)
                ]
                probe_successes = sum(item["success"] for item in probe_rows)
                probe_rate = probe_successes / max(1, len(probe_rows))
                threshold_met = probe_rate >= float(run_config["curriculum_success_threshold"])
                promoted = threshold_met and not bool(run_config.get("pilot_mode", False))
                old_stage = curriculum_stage
                if promoted:
                    curriculum_stage = min(curriculum_stage + 1, len(CURRICULUM_STAGES) - 1)
                curriculum_row = {
                    "episode": episode,
                    "stage_before": old_stage,
                    "stage_before_name": CURRICULUM_STAGES[old_stage]["name"],
                    "probe_difficulty": probe_difficulty,
                    "probe_episodes": len(probe_rows),
                    "probe_successes": probe_successes,
                    "probe_success_rate": probe_rate,
                    "required_success_rate": float(run_config["curriculum_success_threshold"]),
                    "threshold_met": threshold_met,
                    "promotion_suppressed_for_pilot": bool(run_config.get("pilot_mode", False) and threshold_met),
                    "promoted": promoted,
                    "stage_after": curriculum_stage,
                    "stage_after_name": CURRICULUM_STAGES[curriculum_stage]["name"],
                    "probe_terrain_seeds": [item["terrain_seed"] for item in probe_rows],
                }
                append_jsonl(results_dir / "curriculum_log.jsonl", curriculum_row)
                append_jsonl(results_dir / "evaluation_detail.jsonl", {
                    "training_episode": episode,
                    "eval_type": "curriculum_probe",
                    "summary": curriculum_row,
                    "episodes": probe_rows,
                })
                print(
                    f"  Curriculum probe: {probe_difficulty}={probe_rate:.0%} "
                    f"({probe_successes}/{len(probe_rows)}) -> "
                    f"{'PROMOTED' if promoted else 'stay at current stage'}"
                    + (" (pilot: promotion disabled)" if run_config.get("pilot_mode", False) else "")
                )

            if episode % int(run_config["resume_interval"]) == 0 or episode == episodes:
                resume_path = checkpoints_dir / "latest_resume.pt"
                save_training_resume(
                    resume_path,
                    agent=agent,
                    episode=episode,
                    global_step=global_step,
                    curriculum_stage=curriculum_stage,
                    training_rng=training_rng,
                    curriculum_rng=curriculum_rng,
                    best_key=best_key,
                    best_summary=best_summary,
                    run_id=run_id,
                    results_dir=results_dir,
                    checkpoints_dir=checkpoints_dir,
                    run_config=run_config,
                )
                print(f"  Full resume state: {resume_path.relative_to(PROJECT_ROOT)}")

        print("\nPhase 10 full-terrain launch-window DQN training run finished.")
        print("Run ID:", run_id)
        print("Training log:", training_log.relative_to(PROJECT_ROOT))
        print("Evaluation log:", (results_dir / "evaluation_log.csv").relative_to(PROJECT_ROOT))
        print("Terrain snapshots:", (results_dir / "terrain_snapshots").relative_to(PROJECT_ROOT))
        print("Checkpoint folder:", checkpoints_dir.relative_to(PROJECT_ROOT))
        print("Device:", device, "| CUDA available:", torch.cuda.is_available())
        if best_summary is not None:
            print("Best saved validation success:", f"{best_summary['overall_success_rate']:.1%}")
        print("Note: this run's metrics are the measured result; no performance is assumed in advance.")
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VoidRunner Phase 11 transfer-learning DQN training on full terrain")
    parser.add_argument("--episodes", type=int, default=None, help="total episode target; default is 20 for a safe first run")
    parser.add_argument("--max-steps", type=int, default=None, help="step limit per episode (new runs default to 1200)")
    parser.add_argument("--seed", type=int, default=None, help="training seed (new runs default to 42)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None, help="new run device; defaults to auto")
    parser.add_argument("--run-id", default=None, help="optional unique run name for a new run")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="weights-only initialization from a successful single-gap latest.pt; starts a fresh full-terrain run")
    parser.add_argument("--resume", type=Path, default=None, help="full latest_resume.pt checkpoint from this training script")
    parser.add_argument("--pilot", action="store_true", help="run a short 5-episode pipeline test with smaller replay settings")
    parser.add_argument("--render", action="store_true", help="render the training environment (slower; not recommended for normal runs)")
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--min-replay-size", type=int, default=None)
    parser.add_argument("--replay-capacity", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--target-update-frequency", type=int, default=None)
    parser.add_argument("--epsilon-start", type=float, default=None)
    parser.add_argument("--epsilon-end", type=float, default=None)
    parser.add_argument("--epsilon-decay-steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-episodes-per-difficulty", type=int, default=None)
    parser.add_argument("--curriculum-check-interval", type=int, default=None)
    parser.add_argument("--probe-episodes", type=int, default=None)
    parser.add_argument("--curriculum-success-threshold", type=float, default=None)
    parser.add_argument("--resume-interval", type=int, default=None)
    parser.add_argument("--jump-launch-reward", type=float, default=None, help="Training-only reward for a physical Hop/Leap launch near an obstacle (default 2.5).")
    parser.add_argument("--airborne-obstacle-reward", type=float, default=None, help="Training-only reward per airborne step safely clearing a targeted gap/rock (default 0.25).")
    parser.add_argument("--gap-clearance-reward", type=float, default=None, help="Training-only one-off reward for landing beyond a gap after airborne crossing (default 7.0).")
    parser.add_argument("--rock-clearance-reward", type=float, default=None, help="Training-only one-off reward for landing beyond a rock after airborne crossing (default 5.0).")
    parser.add_argument("--approach-distance", type=float, default=None, help="Maximum horizontal distance in pixels for a jump-launch event (default 190).")
    parser.add_argument("--runup-sprint-reward", type=float, default=None, help="One-time training-only reward for reaching high speed in the approach zone (default 0.4).")
    parser.add_argument("--missed-launch-penalty", type=float, default=None, help="Training-only one-time penalty for passing the launch window grounded (default 1.25).")
    parser.add_argument("--early-jump-penalty", type=float, default=None, help="Training-only penalty for launching too far away (default 0.25).")
    parser.add_argument("--late-jump-penalty", type=float, default=None, help="Training-only penalty for launching inside the late zone (default 0.75).")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_training(args)
    except KeyboardInterrupt:
        print("\nInterrupted. The most recent periodic resume checkpoint is preserved if one was saved.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
