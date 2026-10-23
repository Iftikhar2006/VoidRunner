
from environment.terrain_env import TerrainEnv


def test_gap(gap_width):
    env = TerrainEnv(difficulty="easy")

    try:
        env.reset(seed=42)

        # Create one gap with a known width.
        env.gaps = [(200, gap_width)]

        # Start on safe ground before the gap.
        env.player_x = 100.0
        env.player_y = float(
            env.ground_y - env.player_size
        )
        env.vy = 0.0
        env.speed = env.max_speed

        # Move forward and jump when approaching the gap.
        jumped = False
        terminated = False
        truncated = False
        info = {}

        for step in range(200):
            gap_start, _ = env.gaps[0]

            # Jump when the player is close to the gap.
            if (
                not jumped
                and env.player_x + env.player_size
                >= gap_start - 50
            ):
                action = 2
                jumped = True
            else:
                action = 1

            _, _, terminated, truncated, info = env.step(
                action
            )

            if terminated or truncated:
                break

            # Stop after the player has crossed the gap
            # and landed on solid ground.
            if (
                jumped
                and env.player_x > gap_start + gap_width + 40
                and env.vy == 0
                and env.player_y
                == env.ground_y - env.player_size
            ):
                break

        success = (
            not terminated
            and not truncated
            and jumped
            and env.player_x > gap_start + gap_width + 40
            and env.vy == 0
            and env.player_y
            == env.ground_y - env.player_size
        )

        print(
            f"Gap width: {gap_width:3} px | "
            f"{'CROSSED' if success else 'FAILED'} | "
            f"Final x: {env.player_x:.1f} | "
            f"Info: {info}"
        )

    finally:
        env.close()


if __name__ == "__main__":
    for width in (40, 60, 80, 100):
        test_gap(width)
