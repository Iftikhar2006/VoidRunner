
from environment.terrain_env import TerrainEnv


def main():
    env = TerrainEnv(difficulty="medium")

    try:
        observation, info = env.reset(seed=42)

        print("Generated gaps:")
        for start, width in env.gaps:
            print(
                f"  Start: {start}, "
                f"End: {start + width}, "
                f"Width: {width}"
            )

        print("\nJump diagnostics:")

        terminated = False
        truncated = False
        steps = 0
        jump_count = 0

        while not terminated and not truncated and steps < 2000:
            next_gap = next(
                (
                    (start, width)
                    for start, width in env.gaps
                    if start + width > env.player_x
                ),
                None,
            )

            action = 1

            if next_gap is not None:
                gap_start, gap_width = next_gap
                distance = gap_start - env.player_x

                if (
                    env.vy == 0
                    and 0 <= distance <= 65
                ):
                    action = 2

            if action == 2:
                jump_count += 1
                print(
                    f"Jump {jump_count}: "
                    f"x={env.player_x:.1f}, "
                    f"gap_start={gap_start}, "
                    f"gap_width={gap_width}"
                )

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            if action == 2 or terminated or truncated:
                print(
                    f"  After step: "
                    f"x={env.player_x:.1f}, "
                    f"y={env.player_y:.1f}, "
                    f"vy={env.vy:.1f}, "
                    f"terminated={terminated}, "
                    f"info={info}"
                )

            steps += 1

        print("\nFinal results:")
        print("Steps:", steps)
        print("Jumps attempted:", jump_count)
        print("Player position:", env.player_x)
        print("Episode info:", info)

    finally:
        env.close()


if __name__ == "__main__":
    main()
