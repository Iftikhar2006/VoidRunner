"""VoidRunner's side-view robot traversal environment.

Phase 2 implementation for the existing VoidRunner project.

Actions (Discrete(5)):
    0 = brake
    1 = walk
    2 = sprint
    3 = hop
    4 = leap

Observation (Box, shape=(18,)) is documented in OBSERVATION_FEATURES below.

This is a simplified high-level movement simulator, not a joint-level robot
physics model. Coordinates are world-space pixels and one step is one physics
frame. Rendering is optional and does not participate in state transitions.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.terrain_generator import (
    generate_layout,
    load_layout,
    validate_layout,
)


ENVIRONMENT_VERSION = "voidrunner_env_v2"

ACTION_NAMES = {
    0: "brake",
    1: "walk",
    2: "sprint",
    3: "hop",
    4: "leap",
}

# Keep this ordered list synchronized with _get_observation().
OBSERVATION_FEATURES = (
    "horizontal_speed_fraction",
    "vertical_velocity_fraction",  # signed; in [-1, 1]
    "grounded",
    "remaining_energy_fraction",
    "route_progress_fraction",
    "surface_normal_one_hot",
    "surface_rubble_one_hot",
    "surface_mud_one_hot",
    "surface_hazard_one_hot",
    "next_obstacle_distance_fraction",
    "next_obstacle_is_gap",
    "next_obstacle_is_rock",
    "next_obstacle_size_fraction",
    "landing_clearance_after_next_obstacle_fraction",
    "second_obstacle_distance_fraction",
    "second_obstacle_is_gap",
    "second_obstacle_is_rock",
    "second_obstacle_size_fraction",
)

SURFACE_TYPES = ("normal", "rubble", "mud", "hazard")


class TerrainEnv(gym.Env):
    """2D side-view platformer environment for energy-aware robot traversal.

    Parameters
    ----------
    difficulty:
        One of ``"easy"``, ``"medium"``, or ``"hard"``.
    render_mode:
        ``None`` for headless use or ``"human"`` for a Pygame window.
    start_x, goal_x:
        World-space x coordinates, in pixels. The robot starts at start_x and
        succeeds when its right edge reaches goal_x.
    max_steps:
        Maximum physics frames per episode. Reaching it truncates the episode.
    initial_energy:
        Battery budget in abstract energy units. It is normalized in the
        observation but the raw value is reported in info.
    sensor_range:
        Maximum look-ahead range used to normalize obstacle distances.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    # Action indices are part of the external interface; do not reorder them.
    BRAKE = 0
    WALK = 1
    SPRINT = 2
    HOP = 3
    LEAP = 4

    SCREEN_WIDTH = 900
    SCREEN_HEIGHT = 500
    GROUND_Y = 390
    PLAYER_WIDTH = 28
    PLAYER_HEIGHT = 36

    MAX_SPEED = 7.0
    MAX_VERTICAL_SPEED = 16.0
    MAX_GAP_WIDTH = 100.0
    MAX_ROCK_WIDTH = 60.0
    MAX_ROCK_HEIGHT = 55.0
    MAX_CLEARANCE = 450.0

    GRAVITY = 0.65
    HOP_VELOCITY = -10.0
    LEAP_VELOCITY = -13.2

    WALK_SPEED = 3.2
    SPRINT_SPEED = 7.0
    HORIZONTAL_ACCELERATION = 0.30
    BRAKE_DECELERATION = 0.65
    AIR_DRAG = 0.015
    AIR_ACCELERATION = 0.08

    # Surface effects intentionally affect both movement and energy use.
    SURFACE_SPEED_MULTIPLIER = {
        "normal": 1.00,
        "rubble": 0.78,
        "mud": 0.56,
        "hazard": 0.82,
    }
    SURFACE_ENERGY_MULTIPLIER = {
        "normal": 1.00,
        "rubble": 1.25,
        "mud": 1.60,
        "hazard": 1.80,
    }

    # Energy cost per physics frame, before terrain multipliers.
    ACTION_ENERGY_COST = {
        BRAKE: 0.012,
        WALK: 0.030,
        SPRINT: 0.105,
        HOP: 0.025,
        LEAP: 0.030,
    }
    HOP_LAUNCH_ENERGY = 2.5
    LEAP_LAUNCH_ENERGY = 5.5

    PROGRESS_REWARD_WEIGHT = 1.0
    ENERGY_PENALTY_WEIGHT = 0.5
    STEP_PENALTY = 0.001
    HAZARD_CONTACT_PENALTY = 0.001
    GOAL_REWARD = 5.0
    FAILURE_PENALTY = -5.0

    # Actual obstacle sizes are smaller than this observation normalization cap.
    _OBS_LOW = np.asarray(
        [0.0, -1.0, 0.0, 0.0, 0.0] + [0.0] * 13,
        dtype=np.float32,
    )
    _OBS_HIGH = np.ones(18, dtype=np.float32)

    def __init__(
        self,
        difficulty: str = "easy",
        render_mode: str | None = None,
        start_x: int = 100,
        goal_x: int = 2400,
        max_steps: int = 1200,
        initial_energy: float = 100.0,
        sensor_range: float = 450.0,
    ) -> None:
        super().__init__()

        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError("difficulty must be 'easy', 'medium', or 'hard'")
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        if goal_x <= start_x + 320:
            raise ValueError("goal_x must be at least 321 pixels after start_x")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if initial_energy <= 0:
            raise ValueError("initial_energy must be positive")
        if sensor_range <= 0:
            raise ValueError("sensor_range must be positive")

        self.difficulty = difficulty
        self.render_mode = render_mode
        self.start_x = int(start_x)
        self.goal_x = int(goal_x)
        self.max_steps = int(max_steps)
        self.initial_energy = float(initial_energy)
        self.sensor_range = float(sensor_range)

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=self._OBS_LOW.copy(),
            high=self._OBS_HIGH.copy(),
            dtype=np.float32,
        )

        # Geometry is populated by reset().
        self.layout: dict[str, Any] = {}
        self.terrain_seed: int | None = None
        self.gaps: list[tuple[int, int]] = []
        self.surface_patches: list[tuple[int, int, str]] = []
        self.rocks: list[tuple[int, int, int]] = []

        self.x = float(self.start_x)
        self.y = float(self.GROUND_Y - self.PLAYER_HEIGHT)
        self.speed = 0.0
        self.vy = 0.0
        self.grounded = True
        self.energy = self.initial_energy
        self.total_energy_consumed = 0.0
        self.steps = 0
        self.episode_return = 0.0
        self.success = False
        self.failure_reason: str | None = None
        self._episode_done = False
        self._pygame = None
        self._screen = None
        self._clock = None
        self._font = None
        self._small_font = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode.

        Reproducibility:
        - reset(seed=N) reproduces the generated terrain and initial state.
        - reset(options={"terrain_seed": N}) selects an explicit layout seed.
        - reset(options={"layout": layout_dict}) uses the exact supplied layout.
        - reset(options={"layout_path": path}) restores a JSON snapshot.
        """
        super().reset(seed=seed)
        options = options or {}

        if "layout" in options and "layout_path" in options:
            raise ValueError("Provide either options['layout'] or options['layout_path'], not both")

        if "layout_path" in options:
            chosen_layout = load_layout(options["layout_path"])
            layout_source = "snapshot"
        elif "layout" in options:
            chosen_layout = copy.deepcopy(options["layout"])
            # JSON-decoded layouts may contain lists; validate_layout accepts them.
            errors = validate_layout(chosen_layout)
            if errors:
                raise ValueError("Invalid supplied terrain layout: " + "; ".join(errors))
            chosen_layout["gaps"] = [(int(x), int(w)) for x, w in chosen_layout["gaps"]]
            chosen_layout["surface_patches"] = [
                (int(x), int(w), str(kind))
                for x, w, kind in chosen_layout["surface_patches"]
            ]
            chosen_layout["rocks"] = [
                (int(x), int(w), int(h)) for x, w, h in chosen_layout["rocks"]
            ]
            layout_source = "provided_layout"
        else:
            if "terrain_seed" in options:
                chosen_seed = int(options["terrain_seed"])
            else:
                chosen_seed = int(self.np_random.integers(0, 2**32, dtype=np.uint64))
            chosen_layout = generate_layout(
                difficulty=self.difficulty,
                seed=chosen_seed,
                start_x=self.start_x,
                goal_x=self.goal_x,
            )
            layout_source = "generated"

        if int(chosen_layout["start_x"]) != self.start_x:
            raise ValueError(
                f"Layout start_x={chosen_layout['start_x']} does not match environment start_x={self.start_x}"
            )
        if int(chosen_layout["goal_x"]) != self.goal_x:
            raise ValueError(
                f"Layout goal_x={chosen_layout['goal_x']} does not match environment goal_x={self.goal_x}"
            )
        if chosen_layout.get("difficulty", self.difficulty) != self.difficulty:
            # Fixed layouts are allowed across difficulties only when explicitly requested.
            if not options.get("allow_difficulty_mismatch", False):
                raise ValueError(
                    f"Layout difficulty={chosen_layout.get('difficulty')} does not match environment difficulty={self.difficulty}"
                )

        self.layout = chosen_layout
        self.terrain_seed = int(chosen_layout.get("seed", -1))
        self.gaps = list(chosen_layout["gaps"])
        self.surface_patches = list(chosen_layout["surface_patches"])
        self.rocks = list(chosen_layout["rocks"])

        self.x = float(self.start_x)
        self.y = float(self.GROUND_Y - self.PLAYER_HEIGHT)
        self.speed = 0.0
        self.vy = 0.0
        self.grounded = True
        self.energy = self.initial_energy
        self.total_energy_consumed = 0.0
        self.steps = 0
        self.episode_return = 0.0
        self.success = False
        self.failure_reason = None
        self._episode_done = False

        observation = self._get_observation()
        info = {
            "environment_version": ENVIRONMENT_VERSION,
            "difficulty": self.difficulty,
            "terrain_seed": self.terrain_seed,
            "layout_source": layout_source,
            "terrain_layout": self.get_layout_snapshot(),
            "energy_remaining": self.energy,
            "energy_consumed": 0.0,
            "distance_reached": 0.0,
            "progress_fraction": 0.0,
            "steps": 0,
            "success": False,
        }
        return observation, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode_done:
            raise RuntimeError("Episode has ended. Call reset() before calling step() again.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected an integer in [0, 4]")

        action = int(action)
        self._last_action_name = ACTION_NAMES[action]
        self.steps += 1
        previous_progress = self._progress_fraction()
        previous_x = self.x
        previous_y = self.y
        previous_bottom = previous_y + self.PLAYER_HEIGHT
        surface_before = self._surface_at(self.x + self.PLAYER_WIDTH / 2.0)
        was_grounded = self.grounded
        launched_jump = False

        # A jump launches only on the ground. Repeated jump commands in the air
        # cannot reset vertical velocity or create a second jump.
        if was_grounded and action == self.HOP:
            self.vy = self.HOP_VELOCITY
            self.grounded = False
            launched_jump = True
        elif was_grounded and action == self.LEAP:
            self.vy = self.LEAP_VELOCITY
            self.grounded = False
            launched_jump = True

        # Horizontal dynamics use the surface under the robot at the start of
        # this frame. Terrain modifiers do not affect airborne speed control.
        if was_grounded:
            speed_multiplier = self.SURFACE_SPEED_MULTIPLIER[surface_before]
            mobility = speed_multiplier
        else:
            speed_multiplier = 1.0
            mobility = 1.0

        self._apply_horizontal_action(action, mobility, was_grounded, launched_jump)

        # Calculate energy used for this frame, including one-time launch cost.
        raw_energy_cost = self.ACTION_ENERGY_COST[action]
        if launched_jump:
            raw_energy_cost += (
                self.HOP_LAUNCH_ENERGY if action == self.HOP else self.LEAP_LAUNCH_ENERGY
            )
        if was_grounded:
            raw_energy_cost *= self.SURFACE_ENERGY_MULTIPLIER[surface_before]
        energy_spent = min(self.energy, raw_energy_cost)
        self.energy = max(0.0, self.energy - energy_spent)
        self.total_energy_consumed += energy_spent

        # Advance x and vertical physics. Speed is always non-negative; the
        # robot's intended route is forward, not left-right platforming.
        self.x += self.speed
        self.x = min(self.x, float(self.goal_x))

        if self.grounded:
            # If the centre leaves solid ground, the robot begins falling.
            if self._is_gap_at(self.x + self.PLAYER_WIDTH / 2.0):
                self.grounded = False
                self.vy = 0.0
            else:
                self.y = float(self.GROUND_Y - self.PLAYER_HEIGHT)
                self.vy = 0.0
        if not self.grounded:
            self.vy = min(self.MAX_VERTICAL_SPEED, self.vy + self.GRAVITY)
            self.y += self.vy
            current_bottom = self.y + self.PLAYER_HEIGHT
            # A landing is possible only when the feet cross the ground plane
            # from above during this frame. Once the robot has fallen below a
            # gap's platform height, moving horizontally past the gap cannot
            # magically place it back on the platform.
            crossed_ground_from_above = (
                self.vy >= 0
                and previous_bottom <= self.GROUND_Y
                and current_bottom >= self.GROUND_Y
            )
            if crossed_ground_from_above:
                center_x = self.x + self.PLAYER_WIDTH / 2.0
                if not self._is_gap_at(center_x):
                    self.y = float(self.GROUND_Y - self.PLAYER_HEIGHT)
                    self.vy = 0.0
                    self.grounded = True

        # Collision checks are made after the movement update so a jump that
        # actually clears the obstacle is allowed. Rocks are solid, terminal
        # hazards in this high-level model; they are not decorative scenery.
        collided_rock = self._colliding_rock()
        fell = self.y > self.SCREEN_HEIGHT + 15
        reached_goal = self.x + self.PLAYER_WIDTH >= self.goal_x
        exhausted_energy = self.energy <= 0.0

        terminated = False
        truncated = False
        terminal_reward = 0.0
        if reached_goal:
            self.success = True
            self.failure_reason = None
            terminated = True
            terminal_reward = self.GOAL_REWARD
        elif collided_rock is not None:
            self.failure_reason = "rock_collision"
            terminated = True
            terminal_reward = self.FAILURE_PENALTY
        elif fell:
            self.failure_reason = "fell_into_gap"
            terminated = True
            terminal_reward = self.FAILURE_PENALTY
        elif exhausted_energy:
            self.failure_reason = "energy_depleted"
            terminated = True
            terminal_reward = self.FAILURE_PENALTY
        elif self.steps >= self.max_steps:
            self.failure_reason = "time_limit"
            truncated = True

        new_progress = self._progress_fraction()
        progress_fraction_delta = max(0.0, new_progress - previous_progress)
        progress_reward = self.PROGRESS_REWARD_WEIGHT * progress_fraction_delta
        energy_penalty = -self.ENERGY_PENALTY_WEIGHT * (energy_spent / self.initial_energy)
        step_penalty = -self.STEP_PENALTY
        hazard_penalty = (
            -self.HAZARD_CONTACT_PENALTY
            if was_grounded and surface_before == "hazard" and not launched_jump
            else 0.0
        )
        reward_components = {
            "progress": float(progress_reward),
            "energy_penalty": float(energy_penalty),
            "step_penalty": float(step_penalty),
            "hazard_penalty": float(hazard_penalty),
            "terminal": float(terminal_reward),
        }
        reward = float(sum(reward_components.values()))
        self.episode_return += reward

        if terminated or truncated:
            self._episode_done = True

        observation = self._get_observation()
        info = {
            "environment_version": ENVIRONMENT_VERSION,
            "difficulty": self.difficulty,
            "terrain_seed": self.terrain_seed,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "steps": self.steps,
            "x": float(self.x),
            "y": float(self.y),
            "speed": float(self.speed),
            "vertical_velocity": float(self.vy),
            "grounded": bool(self.grounded),
            "surface": self._surface_at(self.x + self.PLAYER_WIDTH / 2.0),
            "energy_remaining": float(self.energy),
            "energy_consumed": float(self.total_energy_consumed),
            "distance_reached": float(max(0.0, self.x - self.start_x)),
            "progress_fraction": float(new_progress),
            "episode_return": float(self.episode_return),
            "reward_components": reward_components,
            "success": bool(self.success),
            "failure_reason": self.failure_reason,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        if collided_rock is not None:
            info["collided_rock"] = {
                "x": int(collided_rock[0]),
                "width": int(collided_rock[1]),
                "height": int(collided_rock[2]),
            }
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation and state helpers
    # ------------------------------------------------------------------

    def _progress_fraction(self) -> float:
        route_length = max(1.0, float(self.goal_x - self.start_x))
        return float(np.clip((self.x - self.start_x) / route_length, 0.0, 1.0))

    def _is_gap_at(self, world_x: float) -> bool:
        return any(gap_x <= world_x < gap_x + gap_width for gap_x, gap_width in self.gaps)

    def _surface_at(self, world_x: float) -> str:
        for patch_x, patch_width, kind in self.surface_patches:
            if patch_x <= world_x < patch_x + patch_width:
                return kind
        return "normal"

    def _obstacles(self) -> list[dict[str, Any]]:
        obstacles: list[dict[str, Any]] = []
        for x, width in self.gaps:
            obstacles.append({
                "type": "gap", "x": float(x), "width": float(width),
                "height": 0.0, "end": float(x + width),
            })
        for x, width, height in self.rocks:
            obstacles.append({
                "type": "rock", "x": float(x), "width": float(width),
                "height": float(height), "end": float(x + width),
            })
        return sorted(obstacles, key=lambda obj: (obj["x"], obj["end"]))

    def _upcoming_obstacles(self) -> list[dict[str, Any]]:
        # Keep an obstacle in view until the robot's left edge has passed its end.
        return [
            obstacle for obstacle in self._obstacles()
            if obstacle["end"] >= self.x - 1e-6
        ]

    def _distance_fraction(self, obstacle: dict[str, Any] | None) -> float:
        if obstacle is None:
            return 1.0
        distance = max(0.0, obstacle["x"] - (self.x + self.PLAYER_WIDTH))
        return float(np.clip(distance / self.sensor_range, 0.0, 1.0))

    def _obstacle_size_fraction(self, obstacle: dict[str, Any] | None) -> float:
        if obstacle is None:
            return 0.0
        if obstacle["type"] == "gap":
            return float(np.clip(obstacle["width"] / self.MAX_GAP_WIDTH, 0.0, 1.0))
        # The single rock-size feature summarizes both height and width; the
        # height is the more important traversal constraint, hence its weighting.
        width_fraction = obstacle["width"] / self.MAX_ROCK_WIDTH
        height_fraction = obstacle["height"] / self.MAX_ROCK_HEIGHT
        return float(np.clip(0.35 * width_fraction + 0.65 * height_fraction, 0.0, 1.0))

    def _landing_clearance_fraction(
        self,
        first: dict[str, Any] | None,
        second: dict[str, Any] | None,
    ) -> float:
        if first is None:
            return 1.0
        next_x = second["x"] if second is not None else float(self.goal_x)
        clearance = max(0.0, next_x - first["end"])
        return float(np.clip(clearance / self.MAX_CLEARANCE, 0.0, 1.0))

    def _get_observation(self) -> np.ndarray:
        surface = self._surface_at(self.x + self.PLAYER_WIDTH / 2.0)
        surface_one_hot = [1.0 if surface == name else 0.0 for name in SURFACE_TYPES]
        upcoming = self._upcoming_obstacles()
        first = upcoming[0] if upcoming else None
        second = upcoming[1] if len(upcoming) > 1 else None

        first_type = first["type"] if first else None
        second_type = second["type"] if second else None
        observation = np.asarray(
            [
                np.clip(self.speed / self.MAX_SPEED, 0.0, 1.0),
                np.clip(self.vy / self.MAX_VERTICAL_SPEED, -1.0, 1.0),
                1.0 if self.grounded else 0.0,
                np.clip(self.energy / self.initial_energy, 0.0, 1.0),
                self._progress_fraction(),
                *surface_one_hot,
                self._distance_fraction(first),
                1.0 if first_type == "gap" else 0.0,
                1.0 if first_type == "rock" else 0.0,
                self._obstacle_size_fraction(first),
                self._landing_clearance_fraction(first, second),
                self._distance_fraction(second),
                1.0 if second_type == "gap" else 0.0,
                1.0 if second_type == "rock" else 0.0,
                self._obstacle_size_fraction(second),
            ],
            dtype=np.float32,
        )
        # Defensive check: fail loudly during development rather than silently
        # feeding malformed state vectors to DQN.
        if observation.shape != (18,):
            raise RuntimeError(f"Internal observation shape error: {observation.shape}")
        if not np.isfinite(observation).all():
            raise RuntimeError("Environment produced a non-finite observation")
        return np.clip(observation, self._OBS_LOW, self._OBS_HIGH).astype(np.float32)

    def _apply_horizontal_action(
        self,
        action: int,
        mobility: float,
        was_grounded: bool,
        launched_jump: bool,
    ) -> None:
        # Jump buttons initiate vertical motion only. They preserve current
        # momentum for the launch frame; they are not magic forward teleports.
        if action in (self.HOP, self.LEAP):
            if not was_grounded:
                # An airborne jump command cannot relaunch the robot.
                self.speed = max(0.0, self.speed - self.AIR_DRAG)
            return

        if not was_grounded:
            # Air control is deliberately weaker than ground acceleration, so
            # takeoff speed and the timing of the maneuver remain important.
            if action == self.BRAKE:
                self.speed = max(0.0, self.speed - 0.22)
                return
            target_speed = self.WALK_SPEED if action == self.WALK else self.SPRINT_SPEED
            if self.speed < target_speed:
                self.speed = min(target_speed, self.speed + self.AIR_ACCELERATION)
            elif self.speed > target_speed:
                self.speed = max(target_speed, self.speed - self.AIR_ACCELERATION * 0.75)
            return

        if action == self.BRAKE:
            self.speed = max(0.0, self.speed - self.BRAKE_DECELERATION * max(0.7, mobility))
            return

        target_speed = self.WALK_SPEED if action == self.WALK else self.SPRINT_SPEED
        target_speed *= mobility
        acceleration = self.HORIZONTAL_ACCELERATION * max(0.45, mobility)
        if self.speed < target_speed:
            self.speed = min(target_speed, self.speed + acceleration)
        elif self.speed > target_speed:
            # Releasing sprint naturally settles speed rather than changing it instantly.
            self.speed = max(target_speed, self.speed - acceleration * 0.75)

    def _colliding_rock(self) -> tuple[int, int, int] | None:
        player_left = self.x
        player_right = self.x + self.PLAYER_WIDTH
        player_top = self.y
        player_bottom = self.y + self.PLAYER_HEIGHT
        for rock_x, rock_width, rock_height in self.rocks:
            rock_left = float(rock_x)
            rock_right = float(rock_x + rock_width)
            rock_top = float(self.GROUND_Y - rock_height)
            rock_bottom = float(self.GROUND_Y)
            horizontal_overlap = player_left < rock_right and player_right > rock_left
            vertical_overlap = player_top < rock_bottom and player_bottom > rock_top
            if horizontal_overlap and vertical_overlap:
                return (rock_x, rock_width, rock_height)
        return None

    def get_layout_snapshot(self) -> dict[str, Any]:
        """Return a deep-copied layout for exact logging/restoration."""
        return copy.deepcopy(self.layout)

    # ------------------------------------------------------------------
    # Rendering — visual only; this method never advances simulation state.
    # ------------------------------------------------------------------

    def render(self) -> None:
        if self.render_mode != "human":
            return
        self._ensure_pygame()
        pygame = self._pygame
        screen = self._screen
        assert pygame is not None and screen is not None
        pygame.event.pump()

        screen_w = self.SCREEN_WIDTH
        screen_h = self.SCREEN_HEIGHT
        ground_y = self.GROUND_Y
        max_camera_x = max(0.0, float(self.goal_x - screen_w))
        camera_x = float(np.clip(self.x - screen_w * 0.32, 0.0, max_camera_x))

        # Sky and a simple parallax backdrop.
        screen.fill((185, 216, 230))
        pygame.draw.circle(screen, (248, 211, 111), (screen_w - 100, 82), 29)
        hill_color_far = (132, 165, 166)
        hill_color_near = (104, 143, 145)
        far_points = [(-40, ground_y - 95)]
        for px in range(-40, screen_w + 100, 120):
            wx = px + camera_x * 0.28
            peak = ground_y - 92 - int(22 * math.sin(wx * 0.006))
            far_points.append((px, peak))
        far_points.extend([(screen_w + 60, ground_y), (-40, ground_y)])
        pygame.draw.polygon(screen, hill_color_far, far_points)

        near_points = [(-40, ground_y - 38)]
        for px in range(-40, screen_w + 100, 150):
            wx = px + camera_x * 0.48
            peak = ground_y - 42 - int(18 * math.sin(wx * 0.0045 + 1.3))
            near_points.append((px, peak))
        near_points.extend([(screen_w + 60, ground_y), (-40, ground_y)])
        pygame.draw.polygon(screen, hill_color_near, near_points)

        # Continuous solid ground, with each gap removed from the platform.
        left_world = camera_x
        right_world = camera_x + screen_w
        cursor = left_world
        visible_gaps = [
            (gx, gx + gw) for gx, gw in self.gaps
            if gx + gw >= left_world and gx <= right_world
        ]
        for gap_start, gap_end in visible_gaps:
            segment_start = max(cursor, left_world)
            segment_end = min(float(gap_start), right_world)
            if segment_end > segment_start:
                self._draw_ground_segment(segment_start, segment_end, camera_x)
            cursor = max(cursor, float(gap_end))
        if cursor < right_world:
            self._draw_ground_segment(cursor, right_world, camera_x)

        # Pits and hazard-striped edges.
        for gx, gw in self.gaps:
            sx = int(gx - camera_x)
            ex = int(gx + gw - camera_x)
            if ex < 0 or sx > screen_w:
                continue
            pygame.draw.rect(screen, (35, 39, 50), (sx, ground_y, max(1, ex - sx), screen_h - ground_y))
            stripe_color = (224, 177, 54)
            for edge_x in (sx - 9, ex):
                for stripe in range(0, 34, 12):
                    pygame.draw.polygon(
                        screen, stripe_color,
                        [(edge_x, ground_y + stripe),
                         (edge_x + 7, ground_y + stripe),
                         (edge_x + 14, ground_y + stripe + 9),
                         (edge_x + 7, ground_y + stripe + 9)],
                    )
            pygame.draw.line(screen, (220, 75, 67), (sx, ground_y), (sx, ground_y - 7), 3)
            pygame.draw.line(screen, (220, 75, 67), (ex, ground_y), (ex, ground_y - 7), 3)

        # Surface patches have both visible textures and real movement/energy effects.
        terrain_colors = {
            "rubble": ((139, 103, 69), (205, 160, 105)),
            "mud": ((72, 100, 120), (115, 155, 175)),
            "hazard": ((130, 48, 48), (245, 85, 65)),
        }
        for px, width, kind in self.surface_patches:
            sx = int(px - camera_x)
            ex = int(px + width - camera_x)
            if ex < 0 or sx > screen_w:
                continue
            base, accent = terrain_colors[kind]
            pygame.draw.rect(screen, base, (sx, ground_y, max(1, ex - sx), screen_h - ground_y))
            pygame.draw.rect(screen, accent, (sx, ground_y, max(1, ex - sx), 4))
            if kind == "rubble":
                for tx in range(max(0, sx + 5), min(screen_w, ex), 19):
                    pygame.draw.polygon(screen, accent, [(tx, ground_y + 10), (tx + 5, ground_y + 4), (tx + 11, ground_y + 12)])
            elif kind == "mud":
                for tx in range(max(0, sx + 6), min(screen_w, ex), 23):
                    pygame.draw.ellipse(screen, accent, (tx, ground_y + 11, 11, 4))
            else:
                for tx in range(max(0, sx - 10), min(screen_w, ex), 16):
                    pygame.draw.line(screen, accent, (tx, ground_y + 8), (tx + 8, ground_y + 22), 3)

        # Rocks: a collision-relevant object, not merely decoration.
        for rx, rw, rh in self.rocks:
            sx = int(rx - camera_x)
            if sx + rw < 0 or sx > screen_w:
                continue
            top = ground_y - rh
            rock_points = [
                (sx, ground_y), (sx + 3, top + 7),
                (sx + rw // 3, top), (sx + 2 * rw // 3, top + 3),
                (sx + rw - 2, top + 12), (sx + rw, ground_y),
            ]
            pygame.draw.polygon(screen, (92, 88, 84), rock_points)
            pygame.draw.lines(screen, (135, 130, 122), False, rock_points[1:-1], 2)

        # Goal marker and finish banner.
        goal_screen_x = int(self.goal_x - camera_x)
        if -30 <= goal_screen_x <= screen_w + 30:
            pygame.draw.line(screen, (57, 177, 105), (goal_screen_x, ground_y - 105), (goal_screen_x, ground_y), 5)
            pygame.draw.rect(screen, (57, 177, 105), (goal_screen_x - 1, ground_y - 105, 61, 25))
            self._draw_text("GOAL", (goal_screen_x + 5, ground_y - 101), (20, 45, 30), small=True)

        # Compact robot rover with body, sensor, articulated legs and feet.
        robot_screen_x = int(self.x - camera_x)
        robot_top = int(self.y)
        body = pygame.Rect(robot_screen_x + 2, robot_top + 8, self.PLAYER_WIDTH - 4, 17)
        pygame.draw.line(screen, (35, 54, 66), (robot_screen_x + 9, robot_top + 24), (robot_screen_x + 5, robot_top + 34), 4)
        pygame.draw.line(screen, (35, 54, 66), (robot_screen_x + 19, robot_top + 24), (robot_screen_x + 24, robot_top + 34), 4)
        pygame.draw.ellipse(screen, (43, 48, 55), (robot_screen_x + 1, robot_top + 29, 10, 7))
        pygame.draw.ellipse(screen, (43, 48, 55), (robot_screen_x + 18, robot_top + 29, 10, 7))
        pygame.draw.rect(screen, (40, 125, 225), body, border_radius=4)
        pygame.draw.rect(screen, (25, 83, 154), (body.x + 2, body.y + 11, body.width - 4, 4), border_radius=2)
        pygame.draw.rect(screen, (169, 222, 236), (body.x + 5, body.y + 3, 9, 6), border_radius=2)
        pygame.draw.line(screen, (35, 54, 66), (robot_screen_x + 22, robot_top + 8), (robot_screen_x + 25, robot_top + 1), 2)
        pygame.draw.circle(screen, (248, 196, 51), (robot_screen_x + 25, robot_top + 1), 3)
        pygame.draw.circle(screen, (252, 236, 176), (robot_screen_x + 25, robot_top + 1), 1)

        # HUD panel.
        pygame.draw.rect(screen, (22, 35, 49), (15, 14, 310, 104), border_radius=10)
        pygame.draw.rect(screen, (70, 96, 114), (15, 14, 310, 104), 2, border_radius=10)
        self._draw_text(f"VOIDRUNNER  |  {self.difficulty.upper()}", (27, 22), (245, 248, 249))
        action_name = getattr(self, "_last_action_name", "ready")
        self._draw_text(f"Mode: {action_name}    Speed: {self.speed:4.1f}", (27, 43), (220, 231, 235), small=True)
        self._draw_text("ENERGY", (27, 65), (235, 241, 242), small=True)
        bar_x, bar_y, bar_w, bar_h = 91, 68, 210, 12
        pygame.draw.rect(screen, (65, 74, 84), (bar_x, bar_y, bar_w, bar_h), border_radius=5)
        energy_ratio = float(np.clip(self.energy / self.initial_energy, 0.0, 1.0))
        energy_color = (69, 188, 112) if energy_ratio > 0.35 else (230, 163, 58) if energy_ratio > 0.15 else (221, 74, 68)
        pygame.draw.rect(screen, energy_color, (bar_x, bar_y, int(bar_w * energy_ratio), bar_h), border_radius=5)
        self._draw_text(f"{self.energy:5.1f}/{self.initial_energy:.0f}", (bar_x + bar_w + 6, bar_y - 2), (240, 244, 245), small=True)
        progress = self._progress_fraction()
        pygame.draw.rect(screen, (65, 74, 84), (27, 91, 274, 8), border_radius=4)
        pygame.draw.rect(screen, (57, 177, 105), (27, 91, int(274 * progress), 8), border_radius=4)
        self._draw_text(f"Progress {progress * 100:5.1f}%   Step {self.steps}/{self.max_steps}", (27, 101), (220, 231, 235), small=True)

        # State / result text and simple controls legend.
        if self.success:
            status_text = "MISSION COMPLETE"
            status_color = (38, 126, 78)
        elif self.failure_reason:
            status_text = f"MISSION ENDED: {self.failure_reason.replace('_', ' ').upper()}"
            status_color = (145, 43, 43)
        else:
            status_text = f"Next hazard: {self._next_hazard_label()}"
            status_color = (22, 35, 49)
        pygame.draw.rect(screen, status_color, (screen_w - 330, 14, 315, 32), border_radius=7)
        self._draw_text(status_text, (screen_w - 320, 22), (250, 250, 250), small=True)
        pygame.draw.rect(screen, (22, 35, 49), (15, screen_h - 30, 440, 20), border_radius=6)
        self._draw_text("0 Brake   1 Walk   2 Sprint   3 Hop   4 Leap", (24, screen_h - 27), (245, 248, 249), small=True)

        pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(self.metadata["render_fps"])

    def _draw_ground_segment(self, world_start: float, world_end: float, camera_x: float) -> None:
        pygame = self._pygame
        screen = self._screen
        assert pygame is not None and screen is not None
        sx = int(world_start - camera_x)
        ex = int(world_end - camera_x)
        if ex <= sx:
            return
        pygame.draw.rect(screen, (205, 204, 187), (sx, self.GROUND_Y, ex - sx, self.SCREEN_HEIGHT - self.GROUND_Y))
        pygame.draw.line(screen, (132, 130, 111), (sx, self.GROUND_Y), (ex, self.GROUND_Y), 3)
        for tx in range(max(0, sx + 4), min(self.SCREEN_WIDTH, ex), 27):
            pygame.draw.line(screen, (180, 178, 161), (tx, self.GROUND_Y + 12), (tx + 8, self.GROUND_Y + 18), 2)

    def _draw_text(
        self,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int],
        small: bool = False,
    ) -> None:
        pygame = self._pygame
        screen = self._screen
        assert pygame is not None and screen is not None
        font = self._small_font if small else self._font
        if font is None:
            return
        image = font.render(text, True, color)
        screen.blit(image, position)

    def _next_hazard_label(self) -> str:
        upcoming = self._upcoming_obstacles()
        if not upcoming:
            return "none — clear route"
        obstacle = upcoming[0]
        distance = max(0.0, obstacle["x"] - (self.x + self.PLAYER_WIDTH))
        return f"{obstacle['type']} in {distance:.0f}px"

    def _ensure_pygame(self) -> None:
        if self._pygame is not None and self._screen is not None:
            return
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover - depends on local optional dependency
            raise ImportError(
                "Pygame is required for render_mode='human'. Install it with: pip install pygame"
            ) from exc
        pygame.init()
        self._pygame = pygame
        self._screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("VoidRunner — Energy-Aware Robot Traversal")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("segoeui", 16, bold=True)
        self._small_font = pygame.font.SysFont("segoeui", 12)

    def close(self) -> None:
        if self._pygame is not None:
            try:
                if self._screen is not None:
                    self._pygame.display.quit()
                self._pygame.quit()
            finally:
                self._pygame = None
                self._screen = None
                self._clock = None
                self._font = None
                self._small_font = None
