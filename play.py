
import pygame

from environment.terrain_env import TerrainEnv


def create_env(difficulty):
    env = TerrainEnv(
        render_mode="human",
        difficulty=difficulty,
    )
    env.reset()

    print(f"\nStarted {difficulty.upper()} terrain.")
    print("Gaps:", env.gaps)

    return env


def main():
    difficulty = "easy"
    env = create_env(difficulty)

    previous_pressed = set()
    episode_finished = False

    print("\n========== VOIDRUNNER ==========")
    print("RIGHT ARROW : Run forward")
    print("LEFT ARROW  : Brake")
    print("SPACE       : Jump")
    print("R           : Restart level")
    print("1           : Easy terrain")
    print("2           : Medium terrain")
    print("3           : Hard terrain")
    print("ESC         : Quit")
    print("================================\n")

    try:
        while True:
            env.render()

            if env.window is None:
                break

            keys = pygame.key.get_pressed()

            if keys[pygame.K_ESCAPE]:
                break

            special_keys = (
                pygame.K_r,
                pygame.K_1,
                pygame.K_2,
                pygame.K_3,
            )

            currently_pressed = {
                key for key in special_keys if keys[key]
            }

            newly_pressed = currently_pressed - previous_pressed
            previous_pressed = currently_pressed

            if pygame.K_r in newly_pressed:
                env.reset()
                episode_finished = False
                print("\nLevel restarted.")
                continue

            selected_difficulty = None

            if pygame.K_1 in newly_pressed:
                selected_difficulty = "easy"
            elif pygame.K_2 in newly_pressed:
                selected_difficulty = "medium"
            elif pygame.K_3 in newly_pressed:
                selected_difficulty = "hard"

            if selected_difficulty is not None:
                env.close()
                difficulty = selected_difficulty
                env = create_env(difficulty)
                episode_finished = False
                previous_pressed = currently_pressed
                continue

            if episode_finished:
                continue

            if keys[pygame.K_SPACE]:
                action = 2
            elif keys[pygame.K_LEFT]:
                action = 0
            elif keys[pygame.K_RIGHT]:
                action = 1
            else:
                action = 0

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            if terminated or truncated:
                episode_finished = True

                if info.get("success", False):
                    print("\nLEVEL COMPLETED!")
                else:
                    print("\nYou fell! Press R to try again.")

                print("Difficulty:", difficulty)
                print("Final position:", round(env.player_x, 1))
                print("Episode info:", info)

    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
