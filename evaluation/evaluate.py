"""VoidRunner Phase 4: observation-only rule-based baseline.

This module implements a hand-designed controller and evaluates it on fixed,
seeded terrain layouts. It deliberately does not inspect env.layout, env.gaps,
env.rocks, or any other hidden environment state while selecting actions.

Run from the repository root:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --difficulty hard --episodes 50
    python evaluation/evaluate.py --episodes 5 --render-first

The script writes episode metrics, a summary, failed runs, configuration, and
one restorable JSON terrain snapshot per episode under results/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.terrain_env import ACTION_NAMES, ENVIRONMENT_VERSION, OBSERVATION_FEATURES, TerrainEnv
from environment.terrain_generator import GENERATOR_VERSION, generate_layout, save_layout
from agent.dqn import DQNAgent


# Feature contract: keep this mapping synchronized with TerrainEnv's documented
# OBSERVATION_FEATURES. These constants are policy assumptions, not hidden state.
OBS_SPEED = 0
OBS_VERTICAL_VELOCITY = 1
OBS_GROUNDED = 2
OBS_ENERGY = 3
OBS_PROGRESS = 4
OBS_SURFACE_START = 5  # normal, rubble, mud, hazard
OBS_NEXT_DISTANCE = 9
OBS_NEXT_IS_GAP = 10
OBS_NEXT_IS_ROCK = 11
OBS_NEXT_SIZE = 12
OBS_LANDING_CLEARANCE = 13
OBS_SECOND_DISTANCE = 14
OBS_SECOND_IS_GAP = 15
OBS_SECOND_IS_ROCK = 16
OBS_SECOND_SIZE = 17

# These values describe the public observation normalization in TerrainEnv.
# If the environment's sensor range or speed normalization changes, update and
# version this heuristic as well.
SENSOR_RANGE_PIXELS = 450.0
MAX_SPEED_PIXELS_PER_FRAME = 7.0
HEURISTIC_VERSION = "rule_baseline_v1"

# Action indices are the current six-action interface in TerrainEnv.
BRAKE = 0
WALK = 1
SPRINT = 2
HOP = 3
LEAP = 4
REVERSE = 5

ACTION_METRIC_NAMES = {
    BRAKE: "brake",
    WALK: "walk",
    SPRINT: "sprint",
    HOP: "hop",
    LEAP: "leap",
    REVERSE: "reverse",
}

DIFFICULTIES = ("easy", "medium", "hard")
DIFFICULTY_IDS = {name: index for index, name in enumerate(DIFFICULTIES)}


def rule_based_action(observation: np.ndarray) -> int:
    """Choose an action using only the same 18-value observation as the DQN.

    Policy outline:
    - Walk on open, normal terrain to conserve energy.
    - Build/maintain speed when a gap or rock is approaching.
    - Leap across wider gaps or a gap followed closely by a rock; use a hop for
      smaller isolated gaps.
    - Use hops for ordinary rocks, reserving leaps for relatively large rocks.
    - Prefer a hop over a rock if another gap is close behind, to reduce the
      chance of carrying too much flight distance into the following gap.
    - Brake during descent only in the narrow case where a rock is being
      cleared and a following gap is very close.
    - If already reversing, command forward movement to recover.

    The policy does not read the generated map or receive any privileged state.
    It is intentionally transparent and deterministic for fair comparison.
    """
    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape != (18,):
        raise ValueError(f"rule_based_action expects observation shape (18,), got {obs.shape}")
    if not np.isfinite(obs).all():
        raise ValueError("observation contains a non-finite value")

    speed = float(obs[OBS_SPEED]) * MAX_SPEED_PIXELS_PER_FRAME
    vertical_velocity = float(obs[OBS_VERTICAL_VELOCITY])
    grounded = bool(obs[OBS_GROUNDED] >= 0.5)
    energy_fraction = float(np.clip(obs[OBS_ENERGY], 0.0, 1.0))

    surface_values = obs[OBS_SURFACE_START : OBS_SURFACE_START + 4]
    surface_index = int(np.argmax(surface_values))
    surface = ("normal", "rubble", "mud", "hazard")[surface_index]

    distance_1 = float(obs[OBS_NEXT_DISTANCE]) * SENSOR_RANGE_PIXELS
    first_is_gap = bool(obs[OBS_NEXT_IS_GAP] >= 0.5)
    first_is_rock = bool(obs[OBS_NEXT_IS_ROCK] >= 0.5)
    size_1 = float(obs[OBS_NEXT_SIZE])
    landing_clearance = float(obs[OBS_LANDING_CLEARANCE]) * 450.0

    distance_2 = float(obs[OBS_SECOND_DISTANCE]) * SENSOR_RANGE_PIXELS
    second_is_gap = bool(obs[OBS_SECOND_IS_GAP] >= 0.5)
    second_is_rock = bool(obs[OBS_SECOND_IS_ROCK] >= 0.5)

    # Airborne control: walking is cheaper on ordinary jumps, but it gently
    # reduces speed. Preserve forward speed when the current jump must clear a
    # gap-plus-rock combination or a rock immediately before another gap.
    # Do not brake while the robot may still horizontally overlap the rock:
    # this environment reports distance=0 throughout that overlap, so braking
    # there can cause the descending robot to collide with the rock's far edge.
    if not grounded:
        tight_combination = (
            first_is_gap and second_is_rock and landing_clearance < 115.0
        ) or (
            first_is_rock and second_is_gap and distance_2 < 275.0
        )
        if tight_combination and energy_fraction > 0.05:
            return SPRINT
        return WALK

    # A negative speed means that the robot is still reversing. Ask it to move
    # forward again; it will transition through zero according to real physics.
    if speed < -0.1:
        return SPRINT if energy_fraction > 0.08 else WALK

    has_obstacle_1 = first_is_gap or first_is_rock

    # No obstacle is currently described by the sensor features. Walk on normal
    # or rubble ground to save energy; use sprint on slow surfaces when the
    # battery is healthy enough to trade energy for forward progress.
    if not has_obstacle_1:
        if surface in ("mud", "hazard") and energy_fraction > 0.35:
            return SPRINT
        return WALK

    if first_is_gap:
        # The generator preserves an acceleration runway before every gap.
        # Start sprinting before entering it. Jump later than a naive distance-
        # only policy: launching too early makes the robot descend onto the far
        # edge of a rock that follows a gap. At high approach speed, 36 px is a
        # safer launch threshold for the current flight physics.
        if distance_1 > 36.0:
            if distance_1 <= 235.0:
                return SPRINT
            return WALK

        # A rock shortly after the gap plus small landing clearance calls for a
        # single longer leap. Otherwise choose the least costly jump that fits.
        gap_plus_rock = second_is_rock and landing_clearance < 115.0
        if gap_plus_rock or size_1 >= 0.58:
            return LEAP
        return HOP

    if first_is_rock:
        # Start preparing before the obstacle rather than trying to jump from a
        # near-zero speed. A hop is normally enough for generated rock sizes.
        if distance_1 > 48.0:
            if distance_1 <= 205.0:
                return SPRINT
            return WALK

        # A short hop is safer when a gap follows soon after the rock; a longer
        # leap is reserved for unusually large rocks or isolated high obstacles.
        if second_is_gap and distance_2 < 275.0 and size_1 < 0.58:
            return HOP
        if size_1 >= 0.58:
            return LEAP
        return HOP

    # Defensive fallback for any valid observation not covered above.
    return WALK


def _mean(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(data) if data else None


def _summary_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row["success"]]
    failure_counts = Counter(
        str(row["failure_reason"] or ("time_limit" if row["truncated"] else "unknown"))
        for row in rows
        if not row["success"]
    )
    return {
        "episodes": len(rows),
        "successes": len(successes),
        "success_rate": len(successes) / len(rows) if rows else 0.0,
        "failure_counts": dict(sorted(failure_counts.items())),
        "mean_return": _mean(row["episode_return"] for row in rows),
        "mean_steps": _mean(row["steps"] for row in rows),
        "mean_completion_seconds": _mean(row["completion_time_seconds"] for row in rows),
        "mean_distance_reached_px": _mean(row["distance_reached_px"] for row in rows),
        "mean_progress_fraction": _mean(row["progress_fraction"] for row in rows),
        "mean_energy_used_all_runs": _mean(row["energy_consumed"] for row in rows),
        "mean_energy_used_successful_runs": _mean(row["energy_consumed"] for row in successes),
        "mean_energy_fraction_used_successful_runs": _mean(
            row["energy_fraction_used"] for row in successes
        ),
        "mean_steps_successful_runs": _mean(row["steps"] for row in successes),
    }


def _make_run_directory(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = parent / timestamp
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_baseline(args: argparse.Namespace) -> Path:
    parent_dir = Path(args.output_dir or "results/phase4_rule_baseline")
    run_dir = _make_run_directory(parent_dir)
    snapshots_dir = run_dir / "terrain_snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    difficulties = list(DIFFICULTIES) if args.difficulty == "all" else [args.difficulty]
    all_rows: list[dict[str, Any]] = []
    per_difficulty: dict[str, Any] = {}
    seed_map: dict[str, list[int]] = {}

    # Capture the exact experiment assumptions so this baseline can be reused
    # later for a fair DQN comparison.
    config = {
        "experiment": "VoidRunner rule-based baseline",
        "heuristic_version": HEURISTIC_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "observation_features": list(OBSERVATION_FEATURES),
        "action_names": {str(k): v for k, v in ACTION_NAMES.items()},
        "episodes_per_difficulty": args.episodes,
        "difficulty_selection": difficulties,
        "seed_start": args.seed_start,
        "seed_policy": "for each difficulty, seed = seed_start + zero_based_episode_index",
        "max_steps": args.max_steps,
        "start_x": 100,
        "goal_x": 2400,
        "initial_energy": 100.0,
        "reward_configuration": {
            "progress_reward_weight": TerrainEnv.PROGRESS_REWARD_WEIGHT,
            "energy_penalty_weight": TerrainEnv.ENERGY_PENALTY_WEIGHT,
            "step_penalty": TerrainEnv.STEP_PENALTY,
            "hazard_contact_penalty": TerrainEnv.HAZARD_CONTACT_PENALTY,
            "goal_reward": TerrainEnv.GOAL_REWARD,
            "failure_penalty": TerrainEnv.FAILURE_PENALTY,
        },
        "controller_limits": {
            "sensor_range_pixels_assumed": SENSOR_RANGE_PIXELS,
            "max_speed_pixels_per_frame_assumed": MAX_SPEED_PIXELS_PER_FRAME,
            "reads_only_observation": True,
            "reads_hidden_layout": False,
        },
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
        file.write("\n")

    render_shown_for: set[str] = set()
    for difficulty in difficulties:
        difficulty_rows: list[dict[str, Any]] = []
        seeds = [args.seed_start + index for index in range(args.episodes)]
        seed_map[difficulty] = seeds

        for episode_index, terrain_seed in enumerate(seeds, start=1):
            should_render = args.render_first and difficulty not in render_shown_for
            render_mode = "human" if should_render else None
            env = TerrainEnv(
                difficulty=difficulty,
                render_mode=render_mode,
                max_steps=args.max_steps,
            )
            action_counts = Counter({name: 0 for name in ACTION_METRIC_NAMES.values()})
            reward_components = Counter()
            total_reward = 0.0
            final_info: dict[str, Any] = {}
            snapshot_relative = Path("terrain_snapshots") / f"{difficulty}_seed_{terrain_seed}.json"
            snapshot_absolute = run_dir / snapshot_relative

            try:
                layout = generate_layout(
                    difficulty=difficulty,
                    seed=terrain_seed,
                    start_x=env.start_x,
                    goal_x=env.goal_x,
                )
                save_layout(layout, snapshot_absolute)
                observation, reset_info = env.reset(
                    seed=terrain_seed,
                    options={"layout": layout},
                )
                if should_render:
                    render_shown_for.add(difficulty)
                    env.render()

                for _ in range(args.max_steps):
                    action = rule_based_action(observation)
                    if not env.action_space.contains(action):
                        raise RuntimeError(f"Controller returned invalid action {action}")
                    action_counts[ACTION_METRIC_NAMES[action]] += 1
                    observation, reward, terminated, truncated, final_info = env.step(action)
                    total_reward += float(reward)
                    for component, value in final_info.get("reward_components", {}).items():
                        reward_components[component] += float(value)
                    if should_render:
                        env.render()
                    if terminated or truncated:
                        break
                else:
                    raise RuntimeError(
                        f"Episode did not terminate or truncate within {args.max_steps} steps"
                    )

                steps = int(final_info.get("steps", env.steps))
                energy_used = float(final_info.get("energy_consumed", env.total_energy_consumed))
                energy_fraction_used = energy_used / float(env.initial_energy)
                row: dict[str, Any] = {
                    "controller": "rule_based",
                    "heuristic_version": HEURISTIC_VERSION,
                    "difficulty": difficulty,
                    "episode_index": episode_index,
                    "terrain_seed": terrain_seed,
                    "environment_version": ENVIRONMENT_VERSION,
                    "generator_version": GENERATOR_VERSION,
                    "success": bool(final_info.get("success", False)),
                    "failure_reason": final_info.get("failure_reason"),
                    "terminated": bool(final_info.get("terminated", False)),
                    "truncated": bool(final_info.get("truncated", False)),
                    "steps": steps,
                    "completion_time_seconds": steps / float(TerrainEnv.metadata.get("render_fps", 60)),
                    "episode_return": float(total_reward),
                    "distance_reached_px": float(final_info.get("distance_reached", 0.0)),
                    "progress_fraction": float(final_info.get("progress_fraction", 0.0)),
                    "energy_initial": float(env.initial_energy),
                    "energy_consumed": energy_used,
                    "energy_remaining": float(final_info.get("energy_remaining", env.energy)),
                    "energy_fraction_used": energy_fraction_used,
                    "snapshot_path": str(snapshot_relative).replace("\\", "/"),
                    "progress_reward_total": float(reward_components.get("progress", 0.0)),
                    "energy_penalty_total": float(reward_components.get("energy_penalty", 0.0)),
                    "step_penalty_total": float(reward_components.get("step_penalty", 0.0)),
                    "hazard_penalty_total": float(reward_components.get("hazard_penalty", 0.0)),
                    "terminal_reward_total": float(reward_components.get("terminal", 0.0)),
                }
                for action_id, metric_name in ACTION_METRIC_NAMES.items():
                    row[f"action_count_{metric_name}"] = int(action_counts[metric_name])
                difficulty_rows.append(row)
                all_rows.append(row)

                if episode_index == 1 or episode_index % max(1, args.episodes // 5) == 0:
                    status = "SUCCESS" if row["success"] else f"FAIL: {row['failure_reason']}"
                    print(
                        f"[{difficulty.upper():6}] episode {episode_index:>3}/{args.episodes} "
                        f"seed={terrain_seed} {status:<20} "
                        f"steps={steps:>4} energy={energy_used:6.2f} "
                        f"progress={row['progress_fraction']:.3f}"
                    )
            finally:
                env.close()

        per_difficulty[difficulty] = _summary_for_rows(difficulty_rows)

    _write_csv(run_dir / "episode_metrics.csv", all_rows)
    _write_csv(run_dir / "failed_episodes.csv", [row for row in all_rows if not row["success"]])

    summary = {
        "experiment": "VoidRunner rule-based baseline",
        "heuristic_version": HEURISTIC_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seeds_by_difficulty": seed_map,
        "by_difficulty": per_difficulty,
        "overall": _summary_for_rows(all_rows),
        "note": (
            "This is a hand-designed baseline, not an RL result. Each controller "
            "decision uses only the observation vector. Terrain snapshots are saved "
            "so later controllers can be evaluated on the same layouts."
        ),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print("\nRULE-BASED BASELINE SUMMARY")
    print("=" * 72)
    for difficulty in difficulties:
        report = per_difficulty[difficulty]
        print(
            f"{difficulty.upper():6}: {report['successes']:>3}/{report['episodes']:<3} "
            f"success ({report['success_rate'] * 100:5.1f}%) | "
            f"mean energy={report['mean_energy_used_all_runs']:.2f} | "
            f"mean steps={report['mean_steps']:.1f}"
        )
        if report["failure_counts"]:
            print(f"        failures: {report['failure_counts']}")
    print(f"\nOverall success rate: {summary['overall']['success_rate'] * 100:.1f}%")
    print(f"Results saved to: {run_dir.resolve()}")
    print("Use the saved terrain_snapshots/*.json files for later DQN comparison.")
    return run_dir



# ---------------------------------------------------------------------------
# Phase 7: fair comparison on identical held-out layouts
# ---------------------------------------------------------------------------

def _phase7_action(
    controller: str,
    observation: np.ndarray,
    *,
    random_rng: np.random.Generator | None = None,
    dqn_agent: DQNAgent | None = None,
) -> int:
    """Select an action using only the standard observation."""
    if controller == "random":
        if random_rng is None:
            raise ValueError("random_rng is required for the random controller")
        return int(random_rng.integers(0, 6))
    if controller == "rule_based":
        return int(rule_based_action(observation))
    if controller.startswith("dqn_"):
        if dqn_agent is None:
            raise ValueError(f"DQN agent is required for controller {controller}")
        return int(dqn_agent.select_action(observation, epsilon=0.0))
    raise ValueError(f"Unknown controller: {controller}")


def _run_phase7_episode(
    *,
    controller: str,
    difficulty: str,
    terrain_seed: int,
    layout: dict[str, Any],
    max_steps: int,
    dqn_agent: DQNAgent | None,
    render: bool,
    snapshot_path: Path,
) -> dict[str, Any]:
    """Evaluate one controller on the supplied exact layout."""
    from environment.terrain_env import TerrainEnv

    env = TerrainEnv(
        difficulty=difficulty,
        render_mode="human" if render else None,
        max_steps=max_steps,
    )
    random_rng = np.random.default_rng((int(terrain_seed) + 91_735) % (2**32 - 1))
    action_counts = Counter({name: 0 for name in ACTION_METRIC_NAMES.values()})
    reward_totals: Counter[str] = Counter()
    total_reward = 0.0
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    try:
        # The seed initializes environment RNG state; the supplied layout
        # ensures every controller receives identical terrain geometry.
        observation, _ = env.reset(
            seed=int(terrain_seed),
            options={"layout": layout},
        )
        if render:
            env.render()
        for _ in range(max_steps):
            action = _phase7_action(
                controller,
                observation,
                random_rng=random_rng,
                dqn_agent=dqn_agent,
            )
            if not env.action_space.contains(action):
                raise RuntimeError(f"{controller} returned invalid action {action}")
            action_counts[ACTION_METRIC_NAMES[action]] += 1
            observation, reward, terminated, truncated, final_info = env.step(action)
            total_reward += float(reward)
            for name, value in final_info.get("reward_components", {}).items():
                reward_totals[str(name)] += float(value)
            if render:
                env.render()
            if terminated or truncated:
                break
        else:
            raise RuntimeError(
                f"{controller} did not terminate or truncate within max_steps={max_steps}"
            )

        steps = int(final_info.get("steps", getattr(env, "steps", max_steps)))
        energy_used = float(final_info.get("energy_consumed", getattr(env, "total_energy_consumed", 0.0)))
        energy_initial = float(getattr(env, "initial_energy", 100.0))
        success = bool(final_info.get("success", False))
        failure_reason = final_info.get("failure_reason")
        if not success and not failure_reason:
            failure_reason = "time_limit" if truncated else "unknown"
        try:
            snapshot_relative = snapshot_path.relative_to(PROJECT_ROOT)
            snapshot_value = str(snapshot_relative).replace("\\", "/")
        except ValueError:
            snapshot_value = str(snapshot_path)
        row: dict[str, Any] = {
            "controller": controller,
            "difficulty": difficulty,
            "terrain_seed": int(terrain_seed),
            "environment_version": ENVIRONMENT_VERSION,
            "generator_version": GENERATOR_VERSION,
            "success": success,
            "failure_reason": failure_reason,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "steps": steps,
            "completion_time_seconds": steps / float(TerrainEnv.metadata.get("render_fps", 60)),
            "episode_return": float(total_reward),
            "distance_reached_px": float(final_info.get("distance_reached", 0.0)),
            "progress_fraction": float(final_info.get("progress_fraction", 0.0)),
            "energy_initial": energy_initial,
            "energy_consumed": energy_used,
            "energy_remaining": float(final_info.get("energy_remaining", getattr(env, "energy", 0.0))),
            "energy_fraction_used": energy_used / max(energy_initial, 1e-9),
            "snapshot_path": snapshot_value,
        }
        for component in ("progress", "energy_penalty", "step_penalty", "hazard_penalty", "terminal"):
            row[f"reward_{component}_total"] = float(reward_totals.get(component, 0.0))
        for action_id, metric_name in ACTION_METRIC_NAMES.items():
            row[f"action_count_{metric_name}"] = int(action_counts[metric_name])
        return row
    finally:
        env.close()


def _summarize_phase7_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if bool(row["success"])]
    failure_counts = Counter(
        str(row["failure_reason"] or "unknown")
        for row in rows if not bool(row["success"])
    )
    return {
        "episodes": len(rows),
        "successes": len(successes),
        "success_rate": len(successes) / len(rows) if rows else 0.0,
        "failure_counts": dict(sorted(failure_counts.items())),
        "mean_return": _mean(row["episode_return"] for row in rows),
        "mean_steps_all_runs": _mean(row["steps"] for row in rows),
        "mean_steps_successful_runs": _mean(row["steps"] for row in successes),
        "mean_completion_seconds_successful_runs": _mean(row["completion_time_seconds"] for row in successes),
        "mean_distance_reached_px": _mean(row["distance_reached_px"] for row in rows),
        "mean_progress_fraction": _mean(row["progress_fraction"] for row in rows),
        "mean_energy_used_all_runs": _mean(row["energy_consumed"] for row in rows),
        "mean_energy_used_successful_runs": _mean(row["energy_consumed"] for row in successes),
    }


def _new_phase7_run_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = parent / base
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{base}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def evaluate_comparison(args: argparse.Namespace) -> Path:
    """Compare random, heuristic, and greedy DQN on identical fixed layouts."""
    import torch
    from environment.terrain_env import TerrainEnv

    if not args.checkpoint:
        raise ValueError("--compare requires --checkpoint pointing to a versioned DQN model checkpoint")
    if args.episodes <= 0 or args.max_steps <= 0 or args.seed_start < 0:
        raise ValueError("episodes/max-steps must be positive and seed-start non-negative")

    device_name = args.device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but torch.cuda.is_available() is False")
    else:
        device = torch.device(device_name)

    selected_difficulties = list(DIFFICULTIES) if args.difficulty == "all" else [args.difficulty]
    # Initialize one environment for feature/action-contract and checkpoint validation.
    probe_env = TerrainEnv(difficulty=selected_difficulties[0], render_mode=None, max_steps=args.max_steps)
    try:
        if tuple(probe_env.observation_space.shape or ()) != (18,) or int(probe_env.action_space.n) != 6:
            raise RuntimeError(
                f"Phase 7 expects observations (18,) and 6 actions; got "
                f"{probe_env.observation_space.shape} and {probe_env.action_space.n}."
            )
        dqn_specs: list[tuple[str, Path]] = [("dqn_best", Path(args.checkpoint))]
        if args.compare_checkpoint:
            dqn_specs.append(("dqn_additional", Path(args.compare_checkpoint)))
        dqn_agents: dict[str, DQNAgent] = {}
        checkpoint_metadata: dict[str, Any] = {}
        for index, (label, checkpoint_path) in enumerate(dqn_specs):
            agent = DQNAgent.from_env(probe_env, device=device, seed=42 + index)
            metadata = agent.load_checkpoint(
                checkpoint_path,
                expected_environment_version=ENVIRONMENT_VERSION,
                load_optimizer=False,
            )
            dqn_agents[label] = agent
            checkpoint_metadata[label] = {
                "path": str(checkpoint_path.resolve()),
                "episode": metadata.get("episode"),
                "global_step": metadata.get("global_step"),
                "learn_steps": metadata.get("learn_steps"),
                "environment_version": metadata.get("environment_version"),
            }
    finally:
        probe_env.close()

    output_parent = Path(args.output_dir or "results/phase7_comparison")
    run_dir = _new_phase7_run_dir(output_parent)
    snapshots_dir = run_dir / "terrain_snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    controllers = ["random", "rule_based", *dqn_agents.keys()]
    seeds_by_difficulty = {
        difficulty: [int(args.seed_start + index) for index in range(args.episodes)]
        for difficulty in selected_difficulties
    }
    config = {
        "experiment": "VoidRunner Phase 7 controller comparison",
        "environment_version": ENVIRONMENT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "observation_features": list(OBSERVATION_FEATURES),
        "action_names": {str(k): v for k, v in ACTION_NAMES.items()},
        "controllers": controllers,
        "dqn_checkpoints": checkpoint_metadata,
        "device": str(device),
        "episodes_per_difficulty": args.episodes,
        "max_steps": args.max_steps,
        "seed_start": args.seed_start,
        "seeds_by_difficulty": seeds_by_difficulty,
        "identical_layouts_for_all_controllers": True,
        "random_controller_seed_rule": "terrain_seed + 91735 (mod 2**32 - 1)",
        "render_first": bool(args.render_first),
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
        file.write("\n")

    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"by_controller": {}, "by_difficulty": {}, "overall": {}}
    rendered_first: set[str] = set()

    for difficulty in selected_difficulties:
        per_diff_rows: list[dict[str, Any]] = []
        for episode_index, terrain_seed in enumerate(seeds_by_difficulty[difficulty], start=1):
            # Generate and persist the exact layout once, then pass the same
            # layout object to every controller's environment reset.
            with_env = TerrainEnv(difficulty=difficulty, render_mode=None, max_steps=args.max_steps)
            try:
                layout = generate_layout(
                    difficulty=difficulty,
                    seed=terrain_seed,
                    start_x=with_env.start_x,
                    goal_x=with_env.goal_x,
                )
            finally:
                with_env.close()
            snapshot_path = snapshots_dir / f"{difficulty}_seed_{terrain_seed}.json"
            save_layout(layout, snapshot_path)

            for controller in controllers:
                render_this = bool(args.render_first and episode_index == 1 and controller not in rendered_first)
                agent = dqn_agents.get(controller)
                row = _run_phase7_episode(
                    controller=controller,
                    difficulty=difficulty,
                    terrain_seed=terrain_seed,
                    layout=layout,
                    max_steps=args.max_steps,
                    dqn_agent=agent,
                    render=render_this,
                    snapshot_path=snapshot_path,
                )
                row["episode_index"] = episode_index
                all_rows.append(row)
                per_diff_rows.append(row)
                if render_this:
                    rendered_first.add(controller)
                if episode_index == 1 or episode_index % max(1, args.episodes // 5) == 0:
                    status = "SUCCESS" if row["success"] else f"FAIL:{row['failure_reason']}"
                    print(
                        f"[{difficulty.upper():6}] {controller:15s} "
                        f"{episode_index:>3}/{args.episodes} seed={terrain_seed} "
                        f"{status:20s} steps={row['steps']:4d} energy={row['energy_consumed']:6.2f}"
                    )
        summary["by_difficulty"][difficulty] = {}
        for controller in controllers:
            rows = [row for row in per_diff_rows if row["controller"] == controller]
            summary["by_difficulty"][difficulty][controller] = _summarize_phase7_rows(rows)

    summary["by_controller"] = {}
    for controller in controllers:
        rows = [row for row in all_rows if row["controller"] == controller]
        summary["by_controller"][controller] = _summarize_phase7_rows(rows)
        summary["by_controller"][controller]["by_difficulty"] = {
            difficulty: summary["by_difficulty"][difficulty][controller]
            for difficulty in selected_difficulties
        }
    summary["overall"] = {
        controller: {
            key: value for key, value in report.items() if key != "by_difficulty"
        }
        for controller, report in summary["by_controller"].items()
    }
    summary["note"] = (
        "Random, rule-based, and DQN controllers were evaluated on identical saved layouts. "
        "The DQN used greedy action selection (epsilon=0). Results are measured data; "
        "no controller is assumed to be superior in advance."
    )

    _write_csv(run_dir / "episode_metrics.csv", all_rows)
    # One concise row per controller/difficulty for easy comparison in Excel.
    comparison_rows: list[dict[str, Any]] = []
    for difficulty in selected_difficulties:
        for controller in controllers:
            report = summary["by_difficulty"][difficulty][controller]
            comparison_rows.append({
                "difficulty": difficulty,
                "controller": controller,
                "episodes": report["episodes"],
                "successes": report["successes"],
                "success_rate": report["success_rate"],
                "mean_return": report["mean_return"],
                "mean_energy_used_all_runs": report["mean_energy_used_all_runs"],
                "mean_energy_used_successful_runs": report["mean_energy_used_successful_runs"],
                "mean_steps_all_runs": report["mean_steps_all_runs"],
                "mean_steps_successful_runs": report["mean_steps_successful_runs"],
                "failure_counts": json.dumps(report["failure_counts"], sort_keys=True),
            })
    _write_csv(run_dir / "comparison_summary.csv", comparison_rows)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print("\nPHASE 7 COMPARISON SUMMARY")
    print("=" * 92)
    print(f"{'Difficulty':<12}{'Controller':<18}{'Success':>12}{'Mean Energy':>15}{'Mean Steps':>13}")
    for difficulty in selected_difficulties:
        for controller in controllers:
            report = summary["by_difficulty"][difficulty][controller]
            energy = report["mean_energy_used_all_runs"]
            steps = report["mean_steps_all_runs"]
            print(
                f"{difficulty:<12}{controller:<18}"
                f"{report['successes']}/{report['episodes']} ({report['success_rate'] * 100:5.1f}%)"
                f"{(energy if energy is not None else float('nan')):>15.2f}"
                f"{(steps if steps is not None else float('nan')):>13.1f}"
            )
    print(f"\nResults: {run_dir.resolve()}")
    print(f"Metrics: {run_dir / 'episode_metrics.csv'}")
    print(f"Summary: {run_dir / 'comparison_summary.csv'}")
    print(f"Exact layouts: {snapshots_dir}")
    return run_dir

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VoidRunner Phase 4 baseline or Phase 7 controller comparison."
    )
    parser.add_argument(
        "--difficulty",
        choices=("easy", "medium", "hard", "all"),
        default="all",
        help="Terrain difficulty to evaluate (default: all).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Number of fixed-seed terrain layouts per selected difficulty (default: 30).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=10000,
        help="First terrain seed; the same seed list is used separately for each difficulty.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1200,
        help="Environment step limit per episode (default: 1200).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Parent directory for a new timestamped result run (mode-specific default if omitted).",
    )
    parser.add_argument(
        "--render-first",
        action="store_true",
        help="Render first evaluation episode(s) for visual inspection.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run Phase 7 comparison: random vs rule-based vs DQN on identical layouts.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Required with --compare: DQN best.pt or other versioned model checkpoint.",
    )
    parser.add_argument(
        "--compare-checkpoint",
        default=None,
        help="Optional second DQN checkpoint (e.g. episode_000500_model.pt) for a matched comparison.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used for DQN evaluation (default: auto).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be a positive integer")
    if args.seed_start < 0:
        parser.error("--seed-start must be non-negative")
    if args.max_steps <= 0:
        parser.error("--max-steps must be a positive integer")
    if args.compare:
        if not args.checkpoint:
            parser.error("--compare requires --checkpoint PATH")
        evaluate_comparison(args)
    else:
        if args.checkpoint or args.compare_checkpoint:
            parser.error("--checkpoint and --compare-checkpoint require --compare")
        evaluate_baseline(args)


if __name__ == "__main__":
    main()
