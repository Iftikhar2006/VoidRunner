
import random


DIFFICULTY_SETTINGS = {
    "easy": {
        "gap_width": (40, 65),
        "gap_spacing": (220, 320),
        "gap_count": (5, 7),
    },
    "medium": {
        "gap_width": (55, 85),
        "gap_spacing": (170, 260),
        "gap_count": (7, 9),
    },
    "hard": {
        "gap_width": (70, 100),
        "gap_spacing": (130, 210),
        "gap_count": (9, 12),
    },
}


def generate_terrain(
    difficulty="easy",
    seed=None,
    start_x=100,
    goal_x=2400,
):
    if difficulty not in DIFFICULTY_SETTINGS:
        raise ValueError(
            "Difficulty must be 'easy', 'medium', or 'hard'."
        )

    rng = random.Random(seed)
    settings = DIFFICULTY_SETTINGS[difficulty]

    min_width, max_width = settings["gap_width"]
    min_spacing, max_spacing = settings["gap_spacing"]
    min_count, max_count = settings["gap_count"]

    gap_count = rng.randint(min_count, max_count)
    gaps = []

    current_x = start_x + rng.randint(180, 260)

    for _ in range(gap_count):
        width = rng.randint(min_width, max_width)

        if current_x + width >= goal_x - 100:
            break

        gaps.append((current_x, width))

        current_x += width + rng.randint(
            min_spacing,
            max_spacing,
        )

    return gaps


if __name__ == "__main__":
    for difficulty in ("easy", "medium", "hard"):
        print(f"\n{difficulty.upper()} TERRAIN")

        gaps = generate_terrain(
            difficulty=difficulty,
            seed=42,
        )

        for index, (start, width) in enumerate(gaps, start=1):
            print(
                f"Gap {index}: "
                f"start={start}, "
                f"width={width}, "
                f"end={start + width}"
            )

        print("Total gaps:", len(gaps))
