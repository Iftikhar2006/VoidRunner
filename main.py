
import pygame

from environment.terrain_env import TerrainEnv


env = TerrainEnv(render_mode="human")

observation, info = env.reset(seed=42)

print("Initial observation:", observation)
print("Observation shape:", observation.shape)
print("Available actions:", env.action_space)
print("Starting test...")

try:
    for step in range(1000):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        env.render()

        if step % 100 == 0:
            print(
                f"Step: {step}, "
                f"Reward: {reward:.2f}, "
                f"Position: {env.player_x:.1f}"
            )

        if terminated or truncated:
            print("Episode ended:", info)
            break

except KeyboardInterrupt:
    print("Test stopped.")

finally:
    env.close()
    pygame.quit()
