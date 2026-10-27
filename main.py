"""VoidRunner entry point: manual play and Phase 3 validation.

Usage from the project root:
    python main.py --manual --difficulty easy --seed 42
    python main.py --validate
    python main.py --validate --render

This file deliberately does not start training. Training integration belongs to
later phases after the environment has passed validation.
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    from environment.terrain_env import ACTION_NAMES, TerrainEnv
except ImportError as exc:  # Give a useful message if an env dependency is missing.
    raise SystemExit(
        "Could not import VoidRunner's environment. From the project root, run:\n"
        "  conda activate voidrunner\n"
        "  pip install gymnasium pygame numpy\n"
        "Original import error: " + str(exc)
    ) from exc


@dataclass
class ValidationResult:
    name: str
    passed: bool
    detail: str = ""


def _empty_layout(
    difficulty: str = "easy",
    start_x: int = 100,
    goal_x: int = 800,
    *,
    gaps: list[tuple[int, int]] | None = None,
    surface_patches: list[tuple[int, int, str]] | None = None,
    rocks: list[tuple[int, int, int]] | None = None,
    seed: int = 123,
) -> dict:
    """Create a minimal valid deterministic test layout.

    This helper is used only for controlled validation cases. It uses the same
    layout interface accepted by TerrainEnv.reset(options={"layout": ...}).
    """
    return {
        "generator_version": 4,
        "seed": int(seed),
        "difficulty": difficulty,
        "start_x": int(start_x),
        "goal_x": int(goal_x),
        "world_width": int(goal_x),
        "coordinate_unit": "pixels",
        "generator_settings": {},
        "generation_clearances": {},
        "gaps": list(gaps or []),
        "surface_patches": list(surface_patches or []),
        "rocks": list(rocks or []),
    }


def _assert_observation(env: TerrainEnv, observation: np.ndarray) -> None:
    assert isinstance(observation, np.ndarray), "observation must be a NumPy array"
    assert observation.shape == (18,), f"expected observation shape (18,), got {observation.shape}"
    assert observation.dtype == np.float32, f"expected float32, got {observation.dtype}"
    assert np.isfinite(observation).all(), "observation contains NaN or infinity"
    assert env.observation_space.contains(observation), "observation is outside observation_space"


def _run_to_end(env: TerrainEnv, action: int, max_iterations: int = 600) -> tuple[bool, bool, dict]:
    """Repeatedly apply a fixed action until termination/truncation or timeout."""
    last_info: dict = {}
    for _ in range(max_iterations):
        observation, reward, terminated, truncated, last_info = env.step(action)
        _assert_observation(env, observation)
        assert math.isfinite(float(reward)), "reward is not finite"
        assert 0.0 <= last_info["energy_remaining"] <= env.initial_energy + 1e-9
        if terminated or truncated:
            return terminated, truncated, last_info
    raise AssertionError(f"episode did not end within {max_iterations} steps")


def _test_gymnasium_checker() -> None:
    from gymnasium.utils.env_checker import check_env

    env = TerrainEnv(render_mode=None)
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def _test_actions_and_observations() -> None:
    env = TerrainEnv(difficulty="easy", render_mode=None, goal_x=800)
    try:
        assert env.action_space.n == 6
        assert env.observation_space.shape == (18,)
        for action in range(6):
            observation, info = env.reset(
                seed=100 + action,
                options={
                    "layout": _empty_layout(
                        difficulty="easy", goal_x=800, seed=100 + action
                    )
                },
            )
            _assert_observation(env, observation)
            next_observation, reward, terminated, truncated, step_info = env.step(action)
            _assert_observation(env, next_observation)
            assert isinstance(reward, (int, float, np.floating))
            assert isinstance(terminated, (bool, np.bool_))
            assert isinstance(truncated, (bool, np.bool_))
            assert step_info["action"] == action
            assert step_info["action_name"] == ACTION_NAMES[action]
            assert 0.0 <= step_info["energy_remaining"] <= env.initial_energy
    finally:
        env.close()


def _test_deterministic_replay() -> None:
    action_sequence = [1] * 18 + [2] * 24 + [3] + [2] * 12 + [4] + [1] * 20 + [5] * 12

    def one_run() -> tuple[dict, list[tuple[np.ndarray, float, bool, bool, float]]]:
        env = TerrainEnv(difficulty="medium", render_mode=None, max_steps=300)
        trace: list[tuple[np.ndarray, float, bool, bool, float]] = []
        try:
            observation, info = env.reset(seed=2026)
            _assert_observation(env, observation)
            saved_layout = info["terrain_layout"]
            for action in action_sequence:
                observation, reward, terminated, truncated, step_info = env.step(action)
                _assert_observation(env, observation)
                trace.append(
                    (
                        observation.copy(),
                        float(reward),
                        bool(terminated),
                        bool(truncated),
                        float(step_info["energy_remaining"]),
                    )
                )
                if terminated or truncated:
                    break
            return saved_layout, trace
        finally:
            env.close()

    layout_a, trace_a = one_run()
    layout_b, trace_b = one_run()
    assert layout_a == layout_b, "same reset seed created different layouts"
    assert len(trace_a) == len(trace_b), "same action sequence ended at different steps"
    for index, (a, b) in enumerate(zip(trace_a, trace_b)):
        assert np.array_equal(a[0], b[0]), f"observation mismatch at trace step {index}"
        assert a[1:] == b[1:], f"reward/terminal/energy mismatch at trace step {index}"


def _test_jump_cannot_relaunch_in_air() -> None:
    env = TerrainEnv(difficulty="easy", render_mode=None, goal_x=800)
    try:
        env.reset(
            seed=3,
            options={"layout": _empty_layout(difficulty="easy", goal_x=800, seed=3)},
        )
        _, _, terminated, truncated, _ = env.step(TerrainEnv.HOP)
        assert not terminated and not truncated
        assert not env.grounded, "hop did not launch from the ground"
        first_vy = env.vy
        assert first_vy < 0, "vertical velocity should be upward after hop"

        _, _, terminated, truncated, _ = env.step(TerrainEnv.LEAP)
        assert not terminated and not truncated
        expected_vy = min(env.MAX_VERTICAL_SPEED, first_vy + env.GRAVITY)
        assert abs(env.vy - expected_vy) < 1e-7, (
            "leap command restarted or changed the jump arc while airborne: "
            f"expected vy={expected_vy}, got {env.vy}"
        )
    finally:
        env.close()



def _test_reverse_action() -> None:
    env = TerrainEnv(difficulty="easy", render_mode=None, goal_x=800)
    try:
        env.reset(
            seed=31,
            options={"layout": _empty_layout(difficulty="easy", goal_x=800, seed=31)},
        )
        for _ in range(22):
            env.step(TerrainEnv.WALK)
        x_before_reverse = env.x
        for _ in range(24):
            observation, reward, terminated, truncated, info = env.step(TerrainEnv.REVERSE)
            if terminated or truncated:
                raise AssertionError(f"reverse test ended unexpectedly: {info}")
        assert env.speed < 0, f"reverse did not produce negative speed: {env.speed}"
        assert env.x < x_before_reverse, (
            f"reverse did not reduce world position: before={x_before_reverse}, after={env.x}"
        )
        assert observation[0] < 0, "signed-speed observation should be negative while reversing"
    finally:
        env.close()


def _test_terminal_conditions() -> None:
    # Goal success on a clear, controlled route.
    goal_env = TerrainEnv(difficulty="easy", render_mode=None, goal_x=500, max_steps=300)
    try:
        goal_env.reset(
            seed=1,
            options={"layout": _empty_layout(difficulty="easy", goal_x=500, seed=1)},
        )
        terminated, truncated, info = _run_to_end(goal_env, TerrainEnv.SPRINT, 150)
        assert terminated and not truncated, f"goal route ended incorrectly: {info}"
        assert info["success"] is True and info["failure_reason"] is None
    finally:
        goal_env.close()

    # Falling into a real gap. The gap is at the first legal location after the
    # environment/generator's protected spawn region.
    fall_env = TerrainEnv(difficulty="easy", render_mode=None, goal_x=800, max_steps=600)
    try:
        fall_layout = _empty_layout(
            difficulty="easy", goal_x=800, gaps=[(320, 60)], seed=2
        )
        fall_env.reset(seed=2, options={"layout": fall_layout})
        terminated, truncated, info = _run_to_end(fall_env, TerrainEnv.SPRINT, 250)
        assert terminated and not truncated, f"gap route did not terminate as failure: {info}"
        assert info["failure_reason"] == "fell_into_gap", f"wrong gap outcome: {info}"
        assert not info["success"]
    finally:
        fall_env.close()

    # Rock collision. The robot does not jump, so it should collide with the rock.
    rock_env = TerrainEnv(difficulty="easy", render_mode=None, goal_x=800, max_steps=600)
    try:
        rock_layout = _empty_layout(
            difficulty="easy", goal_x=800, rocks=[(320, 32, 30)], seed=3
        )
        rock_env.reset(seed=3, options={"layout": rock_layout})
        terminated, truncated, info = _run_to_end(rock_env, TerrainEnv.SPRINT, 250)
        assert terminated and not truncated, f"rock route did not terminate as failure: {info}"
        assert info["failure_reason"] == "rock_collision", f"wrong rock outcome: {info}"
    finally:
        rock_env.close()

    # Battery exhaustion.
    energy_env = TerrainEnv(
        difficulty="easy", render_mode=None, goal_x=800, max_steps=100,
        initial_energy=0.001,
    )
    try:
        energy_env.reset(
            seed=4,
            options={"layout": _empty_layout(difficulty="easy", goal_x=800, seed=4)},
        )
        observation, reward, terminated, truncated, info = energy_env.step(TerrainEnv.WALK)
        assert terminated and not truncated, f"energy exhaustion not terminal: {info}"
        assert info["failure_reason"] == "energy_depleted"
        assert info["energy_remaining"] == 0.0
    finally:
        energy_env.close()

    # Time limit must truncate rather than mark mission failure as termination.
    time_env = TerrainEnv(difficulty="easy", render_mode=None, goal_x=800, max_steps=1)
    try:
        time_env.reset(
            seed=5,
            options={"layout": _empty_layout(difficulty="easy", goal_x=800, seed=5)},
        )
        _, _, terminated, truncated, info = time_env.step(TerrainEnv.WALK)
        assert terminated and not truncated, f"time limit flags are incorrect: {info}"
        assert info["failure_reason"] == "time_limit"
    finally:
        time_env.close()


def _test_invalid_actions_and_step_after_done() -> None:
    env = TerrainEnv(difficulty="easy", render_mode=None, goal_x=500)
    try:
        env.reset(
            seed=7,
            options={"layout": _empty_layout(difficulty="easy", goal_x=500, seed=7)},
        )
        try:
            env.step(6)
        except ValueError:
            pass
        else:
            raise AssertionError("action 6 should be rejected for Discrete(6)")

        env.reset(
            seed=8,
            options={"layout": _empty_layout(difficulty="easy", goal_x=500, seed=8)},
        )
        # Finish the clear course, then make sure step() requires reset().
        terminated = truncated = False
        for _ in range(150):
            _, _, terminated, truncated, _ = env.step(TerrainEnv.SPRINT)
            if terminated or truncated:
                break
        assert terminated or truncated, "controlled route did not end"
        try:
            env.step(TerrainEnv.BRAKE)
        except RuntimeError:
            pass
        else:
            raise AssertionError("step after episode end should raise RuntimeError")
    finally:
        env.close()


def _state_signature(env: TerrainEnv) -> tuple:
    return (
        env.x, env.y, env.speed, env.vy, env.grounded, env.energy,
        env.total_energy_consumed, env.steps, env.episode_return,
    )


def _test_rendering_reinitialization_and_invariance() -> None:
    """Open/render/close twice and verify rendering leaves physics unchanged."""
    for cycle in range(2):
        env = TerrainEnv(difficulty="easy", render_mode="human", goal_x=800)
        try:
            env.reset(
                seed=90 + cycle,
                options={
                    "layout": _empty_layout(
                        difficulty="easy", goal_x=800, seed=90 + cycle
                    )
                },
            )
            env.step(TerrainEnv.WALK)
            before = _state_signature(env)
            env.render()
            after = _state_signature(env)
            assert before == after, "render() changed simulation state"
        finally:
            env.close()


def run_validation(include_render: bool = False) -> int:
    """Run independent validation checks; return a process exit code."""
    checks: list[tuple[str, Callable[[], None]]] = [
        ("Gymnasium environment checker", _test_gymnasium_checker),
        ("Six actions and 18-element observations", _test_actions_and_observations),
        ("Deterministic seeded replay", _test_deterministic_replay),
        ("Jump cannot relaunch while airborne", _test_jump_cannot_relaunch_in_air),
        ("Goal, gap, rock, energy, and time-limit outcomes", _test_terminal_conditions),
        ("Invalid action and post-terminal step handling", _test_invalid_actions_and_step_after_done),
    ]
    if include_render:
        checks.append(("Rendering invariant and repeated open/close", _test_rendering_reinitialization_and_invariance))

    results: list[ValidationResult] = []
    print("\nVOIDRUNNER PHASE 3 VALIDATION")
    print("=" * 46)
    for name, check in checks:
        try:
            check()
            results.append(ValidationResult(name, True))
            print(f"[PASS] {name}")
        except Exception as exc:
            results.append(ValidationResult(name, False, f"{type(exc).__name__}: {exc}"))
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)

    passed = sum(result.passed for result in results)
    failed = len(results) - passed
    print("=" * 46)
    print(f"Result: {passed}/{len(results)} checks passed; {failed} failed.")
    if failed:
        print("Fix every failed check before starting DQN training.")
        return 1
    print("Validation passed. Manually inspect the revised terrain and movement before model integration.")
    return 0


def run_manual(difficulty: str = "easy", seed: int | None = None, max_steps: int = 1200) -> int:
    """Keyboard-controlled manual play using the real TerrainEnv physics."""
    try:
        import pygame
    except ImportError as exc:
        print("Pygame is required for manual mode. Install it with: pip install pygame")
        return 2

    env = TerrainEnv(difficulty=difficulty, render_mode="human", max_steps=max_steps)
    env.manual_mode = True
    try:
        observation, info = env.reset(seed=seed)
        current_layout = info["terrain_layout"]
        current_seed = seed if seed is not None else int(info["terrain_seed"])
        finished = False
        env.manual_last_energy_cost = 0.0
        env.manual_last_action = "ready"
        running = True

        print("VoidRunner manual play")
        print("  A / Left Arrow : reverse / move left")
        print("  S / 0          : brake")
        print("  W / Right Arrow: walk forward")
        print("  D / 2          : sprint")
        print("  Space / 3      : hop (one press)")
        print("  Shift / 4      : leap (one press)")
        print("  R: retry same course | N: new course | Esc: quit")

        # TerrainEnv owns the full frame render, including controls and feedback.
        # There is exactly one display flip per frame, preventing the former HUD flicker.
        env.render()

        while running:
            pulse_action: int | None = None
            did_reset = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        observation, info = env.reset(options={"layout": current_layout})
                        current_layout = info["terrain_layout"]
                        finished = False
                        env.manual_last_energy_cost = 0.0
                        env.manual_last_action = "ready"
                        did_reset = True
                    elif event.key == pygame.K_n:
                        current_seed += 1
                        observation, info = env.reset(seed=current_seed)
                        current_layout = info["terrain_layout"]
                        finished = False
                        env.manual_last_energy_cost = 0.0
                        env.manual_last_action = "ready"
                        did_reset = True
                    elif not finished:
                        if event.key in (pygame.K_3, pygame.K_SPACE):
                            pulse_action = TerrainEnv.HOP
                        elif event.key in (pygame.K_4, pygame.K_LSHIFT, pygame.K_RSHIFT):
                            pulse_action = TerrainEnv.LEAP

            if not running:
                break

            if not finished and not did_reset:
                keys = pygame.key.get_pressed()
                if pulse_action is not None:
                    action = pulse_action
                elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
                    action = TerrainEnv.REVERSE
                elif keys[pygame.K_s] or keys[pygame.K_0]:
                    action = TerrainEnv.BRAKE
                elif keys[pygame.K_w] or keys[pygame.K_RIGHT] or keys[pygame.K_1]:
                    action = TerrainEnv.WALK
                elif keys[pygame.K_d] or keys[pygame.K_2]:
                    action = TerrainEnv.SPRINT
                else:
                    action = TerrainEnv.BRAKE

                energy_before = env.energy
                observation, reward, terminated, truncated, info = env.step(action)
                env.manual_last_action = ACTION_NAMES[action]
                env.manual_last_energy_cost = float(max(0.0, energy_before - env.energy))
                finished = terminated or truncated

            # All text and HUD elements are drawn inside env.render before its
            # single display flip; don't paint another overlay after the flip.
            env.render()

        return 0
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VoidRunner: manual side-view traversal and environment validation."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manual", action="store_true", help="open the side-view game for keyboard control")
    mode.add_argument("--validate", action="store_true", help="run automated environment checks")
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="easy")
    parser.add_argument("--seed", type=int, default=42, help="initial course seed for manual mode")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument(
        "--render",
        action="store_true",
        help="with --validate, include graphical render invariance/reinitialization checks",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_steps <= 0:
        parser.error("--max-steps must be a positive integer")
    if args.render and not args.validate:
        parser.error("--render can only be used with --validate")
    if args.manual:
        return run_manual(args.difficulty, args.seed, args.max_steps)
    return run_validation(include_render=args.render)


if __name__ == "__main__":
    raise SystemExit(main())

