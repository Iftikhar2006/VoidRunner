
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from environment.terrain_generator import generate_terrain


class TerrainEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, difficulty="hard"):
        super().__init__()

        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError(
                "Difficulty must be 'easy', 'medium', or 'hard'."
            )

        self.render_mode = render_mode
        self.difficulty = difficulty

        self.width = 900
        self.height = 500
        self.ground_y = 390
        self.player_size = 30

        self.gravity = 0.7
        self.jump_strength = -12.0
        self.max_speed = 7.0

        self.start_x = 100.0
        self.goal_x = 2400.0

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=np.array(
                [0, 0, 0, -1, 0, 0, 0, 0],
                dtype=np.float32,
            ),
            high=np.ones(8, dtype=np.float32),
            dtype=np.float32,
        )

        self.window = None
        self.clock = None
        self.camera_x = 0.0

        self.gaps = []
        self.reset()

    def _is_over_gap(self, x):
        return any(
            start <= x < start + width
            for start, width in self.gaps
        )

    def _get_obs(self):
        next_gap = next(
            (
                (start, width)
                for start, width in self.gaps
                if start + width > self.player_x
            ),
            (self.goal_x, 0),
        )

        gap_start, gap_width = next_gap
        distance = max(0.0, gap_start - self.player_x)

        grounded = (
            self.player_y >= self.ground_y - self.player_size
            and self.vy == 0
        )

        return np.array(
            [
                min(distance / 500, 1.0),
                min(gap_width / 150, 1.0),
                np.clip(
                    (
                        self.ground_y
                        - self.player_size
                        - self.player_y
                    ) / 150,
                    0,
                    1,
                ),
                np.clip(self.vy / 15, -1, 1),
                float(grounded),
                np.clip(self.speed / self.max_speed, 0, 1),
                np.clip(
                    (self.goal_x - self.player_x) / self.goal_x,
                    0,
                    1,
                ),
                np.clip(self.player_x / self.goal_x, 0, 1),
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.gaps = generate_terrain(
            difficulty=self.difficulty,
            seed=seed,
            start_x=int(self.start_x),
            goal_x=int(self.goal_x),
        )

        self.player_x = self.start_x
        self.player_y = float(
            self.ground_y - self.player_size
        )
        self.vy = 0.0
        self.speed = self.max_speed
        self.camera_x = 0.0
        self.done = False

        return self._get_obs(), {
            "difficulty": self.difficulty,
            "gap_count": len(self.gaps),
        }

    def step(self, action):
        if self.done:
            raise RuntimeError(
                "Episode finished. Call reset() before step()."
            )

        if not self.action_space.contains(action):
            raise ValueError("Action must be 0, 1, or 2.")

        reward = -0.01

        if action == 0:
            self.speed = max(1.0, self.speed - 0.5)

        elif action == 1:
            self.speed = min(
                self.max_speed,
                self.speed + 0.25,
            )

        elif action == 2:
            if (
                self.player_y >= self.ground_y - self.player_size
                and not self._is_over_gap(
                    self.player_x + self.player_size / 2
                )
            ):
                self.vy = self.jump_strength

            self.speed = min(
                self.max_speed,
                self.speed + 0.1,
            )

        previous_x = self.player_x
        self.player_x += self.speed
        reward += (self.player_x - previous_x) * 0.01

        self.vy += self.gravity
        self.player_y += self.vy

        center_x = self.player_x + self.player_size / 2
        over_gap = self._is_over_gap(center_x)

        if (
            self.vy >= 0
            and self.player_y + self.player_size >= self.ground_y
            and not over_gap
        ):
            self.player_y = float(
                self.ground_y - self.player_size
            )
            self.vy = 0.0

        if self.player_y > self.height:
            self.done = True
            return self._get_obs(), -100.0, True, False, {
                "success": False,
                "reason": "fell_into_gap",
            }

        if self.player_x >= self.goal_x:
            self.done = True
            return self._get_obs(), 100.0, True, False, {
                "success": True,
                "reason": "reached_goal",
            }

        return self._get_obs(), reward, False, False, {
            "success": False,
        }

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.width, self.height)
            )
            pygame.display.set_caption("VOIDRUNNER")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.camera_x = max(
            0,
            self.player_x - self.width * 0.3,
        )

        self.window.fill((18, 25, 45))

        previous_end = 0

        for gap_start, gap_width in self.gaps:
            pygame.draw.rect(
                self.window,
                (65, 80, 100),
                (
                    int(previous_end - self.camera_x),
                    self.ground_y,
                    int(gap_start - previous_end),
                    self.height - self.ground_y,
                ),
            )

            for edge_x in (
                gap_start,
                gap_start + gap_width,
            ):
                pygame.draw.line(
                    self.window,
                    (220, 75, 75),
                    (
                        int(edge_x - self.camera_x),
                        self.ground_y,
                    ),
                    (
                        int(edge_x - self.camera_x),
                        self.ground_y + 18,
                    ),
                    4,
                )

            previous_end = gap_start + gap_width

        pygame.draw.rect(
            self.window,
            (65, 80, 100),
            (
                int(previous_end - self.camera_x),
                self.ground_y,
                int(self.goal_x - previous_end),
                self.height - self.ground_y,
            ),
        )

        pygame.draw.rect(
            self.window,
            (70, 210, 230),
            (
                int(self.player_x - self.camera_x),
                int(self.player_y),
                self.player_size,
                self.player_size,
            ),
            border_radius=6,
        )

        pygame.draw.line(
            self.window,
            (90, 230, 130),
            (
                int(self.goal_x - self.camera_x),
                0,
            ),
            (
                int(self.goal_x - self.camera_x),
                self.height,
            ),
            4,
        )

        font = pygame.font.Font(None, 28)
        label = font.render(
            f"VOIDRUNNER - {self.difficulty.upper()} TERRAIN",
            True,
            (255, 255, 255),
        )
        self.window.blit(label, (20, 20))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            self.window = None
            self.clock = None
