"""Deterministic, traversal-aware terrain generation for VoidRunner.

Public interfaces retained:
    generate_terrain(difficulty, seed, start_x, goal_x)
        -> [(gap_start_x, gap_width), ...]
    generate_surface_patches(gaps, difficulty, seed, start_x, goal_x)
        -> [(patch_start_x, patch_width, terrain_type), ...]
    generate_obstacles(gaps, surface_patches, difficulty, seed, start_x, goal_x)
        -> [(rock_start_x, rock_width, rock_height), ...]
    generate_layout(...), validate_layout(...), save_layout(...), load_layout(...)

The new generator profile is deliberately conservative about physical feasibility:
- gaps stay within a range the current leap physics can clear at a sensible speed;
- consecutive gaps leave a landing/repositioning area;
- the final approach to a gap has a protected acceleration runway;
- isolated rocks are smaller than before;
- selected gap-plus-rock combinations are limited to dimensions that can be
  cleared by one well-timed leap under the current environment physics.

This is a physics-informed generation profile, not a formal proof that every
possible policy will clear every course. Use the same-seed environment audit
and evaluation tests to verify actual behavior after changing physics.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

GENERATOR_VERSION = 5
FEASIBILITY_PROFILE_VERSION = 5

# All dimensions are world-space pixels. Spacing is measured from the end of
# one gap to the start of the next gap (solid-ground distance).
DIFFICULTY_SETTINGS: dict[str, dict[str, tuple[int, ...]]] = {
    "easy": {
        "gap_width": (42, 62),
        "gap_spacing": (240, 290),
        "gap_count": (5, 6),
        "patch_count": (5, 8),
        "patch_width": (40, 68),
        "rock_count": (2, 4),
        "rock_width": (22, 36),
        "rock_height": (14, 27),
        "terrain_weights": (5, 4, 1),
        "approach_patch_count": (1, 2),
        "combo_rock_count": (1, 2),
        "combo_rock_gap_offset": (25, 48),
        "combo_rock_width": (22, 30),
        "combo_rock_height": (15, 24),
    },
    "medium": {
        "gap_width": (54, 76),
        "gap_spacing": (235, 270),
        "gap_count": (6, 7),
        "patch_count": (6, 9),
        "patch_width": (42, 68),
        "rock_count": (3, 5),
        "rock_width": (22, 38),
        "rock_height": (18, 31),
        "terrain_weights": (3, 4, 3),
        "approach_patch_count": (2, 3),
        "combo_rock_count": (2, 3),
        "combo_rock_gap_offset": (25, 48),
        "combo_rock_width": (22, 32),
        "combo_rock_height": (18, 27),
    },
    "hard": {
        "gap_width": (65, 88),
        "gap_spacing": (230, 260),
        "gap_count": (6, 7),
        "patch_count": (7, 10),
        "patch_width": (42, 65),
        "rock_count": (4, 6),
        "rock_width": (24, 42),
        "rock_height": (20, 36),
        "terrain_weights": (2, 3, 5),
        "approach_patch_count": (2, 4),
        "combo_rock_count": (3, 4),
        "combo_rock_gap_offset": (25, 48),
        "combo_rock_width": (24, 32),
        "combo_rock_height": (20, 30),
    },
}
TERRAIN_TYPES = ("rubble", "mud", "hazard")

# These limits match the current TerrainEnv's player size, jump velocities,
# acceleration, and collision shape. If those physics change, revisit these.
SPAWN_CLEARANCE = 220
GOAL_CLEARANCE = 100
GAP_APPROACH_RUNUP = 140
GAP_EDGE_CLEARANCE_FOR_PATCH = 12
PATCH_SEPARATION = 25
MIN_GAP_SPACING = 230
MAX_GENERATED_GAP_WIDTH = 88
MAX_GENERATED_ROCK_WIDTH = 42
MAX_GENERATED_ROCK_HEIGHT = 36
GAP_GEOMETRY_CLEARANCE_FOR_ROCK = 12
ISOLATED_ROCK_GAP_CLEARANCE = 85
ROCK_SEPARATION = 75
MAX_PLACEMENT_ATTEMPTS = 500

# Compatibility aliases retained for code that imported the older names.
GAP_CLEARANCE_FOR_PATCH = GAP_EDGE_CLEARANCE_FOR_PATCH
GAP_CLEARANCE_FOR_ROCK = ISOLATED_ROCK_GAP_CLEARANCE


def _check_inputs(difficulty: str, start_x: int, goal_x: int) -> None:
    if difficulty not in DIFFICULTY_SETTINGS:
        allowed = ", ".join(DIFFICULTY_SETTINGS)
        raise ValueError(f"difficulty must be one of: {allowed}")
    if not isinstance(start_x, (int, float)) or not isinstance(goal_x, (int, float)):
        raise TypeError("start_x and goal_x must be numbers")
    if goal_x <= start_x + SPAWN_CLEARANCE + GOAL_CLEARANCE:
        raise ValueError("goal_x must leave enough room for a route")


def _effective_seed(seed: int | None) -> int:
    """Resolve None to an explicit seed so the resulting layout can be replayed."""
    if seed is None:
        return random.SystemRandom().randrange(0, 2**32)
    if not isinstance(seed, int):
        raise TypeError("seed must be an int or None")
    return seed


def _overlaps(
    a_start: float,
    a_width: float,
    b_start: float,
    b_width: float,
    clearance: float = 0,
) -> bool:
    """Return True if intervals overlap, or are closer than clearance."""
    return (
        a_start < b_start + b_width + clearance
        and a_start + a_width + clearance > b_start
    )


def _generate_terrain(
    difficulty: str,
    rng: random.Random,
    start_x: int | float,
    goal_x: int | float,
) -> list[tuple[int, int]]:
    settings = DIFFICULTY_SETTINGS[difficulty]
    min_width, max_width = settings["gap_width"]
    min_spacing, max_spacing = settings["gap_spacing"]
    min_count, max_count = settings["gap_count"]

    target_count = rng.randint(min_count, max_count)
    gaps: list[tuple[int, int]] = []
    current_x = int(start_x + SPAWN_CLEARANCE + rng.randint(20, 60))
    last_safe_edge = int(goal_x - GOAL_CLEARANCE)

    for _ in range(target_count):
        width = rng.randint(min_width, max_width)
        if current_x + width > last_safe_edge:
            break
        gaps.append((current_x, width))
        current_x += width + rng.randint(min_spacing, max_spacing)
    return gaps


def generate_terrain(
    difficulty: str = "easy",
    seed: int | None = None,
    start_x: int = 100,
    goal_x: int = 2400,
) -> list[tuple[int, int]]:
    """Generate feasible-sized gaps as ``(start_x, width)`` tuples."""
    _check_inputs(difficulty, start_x, goal_x)
    effective_seed = _effective_seed(seed)
    return _generate_terrain(
        difficulty, random.Random(effective_seed), start_x, goal_x
    )


def _patch_is_safe(
    x: int,
    width: int,
    gaps: list[tuple[int, int]],
    patches: list[tuple[int, int, str]],
    start_x: int | float,
    goal_x: int | float,
) -> bool:
    if width <= 0 or x < start_x + SPAWN_CLEARANCE or x + width > goal_x - GOAL_CLEARANCE:
        return False

    for gap_x, gap_width in gaps:
        gap_end = gap_x + gap_width
        # Do not draw a patch over a pit or immediately at its edge.
        if _overlaps(x, width, gap_x, gap_width, GAP_EDGE_CLEARANCE_FOR_PATCH):
            return False
        # The ground immediately before every gap is a protected run-up.
        # Patches are allowed earlier in the inter-gap segment, but must end at
        # least GAP_APPROACH_RUNUP pixels before the upcoming gap.
        if x < gap_x and x + width > gap_x - GAP_APPROACH_RUNUP:
            return False
        # A patch may follow a gap only after a small edge clearance. The much
        # more important long clearance applies before the *next* gap above.
        if x >= gap_end and x < gap_end + GAP_EDGE_CLEARANCE_FOR_PATCH:
            return False

    for old_x, old_width, _old_kind in patches:
        if _overlaps(x, width, old_x, old_width, PATCH_SEPARATION):
            return False
    return True


def _generate_surface_patches(
    gaps: Iterable[tuple[int, int]],
    difficulty: str,
    rng: random.Random,
    start_x: int | float,
    goal_x: int | float,
) -> list[tuple[int, int, str]]:
    settings = DIFFICULTY_SETTINGS[difficulty]
    target_count = rng.randint(*settings["patch_count"])
    min_width, max_width = settings["patch_width"]
    gap_list = list(gaps)
    patches: list[tuple[int, int, str]] = []
    weights = settings["terrain_weights"]
    min_x = int(start_x + SPAWN_CLEARANCE)
    max_end = int(goal_x - GOAL_CLEARANCE)

    # First place a limited number of patches on the approach to a gap, but
    # end each patch BEFORE the protected acceleration runway starts. This
    # keeps mud/hazard decisions meaningful without making a gap impossible
    # just because the robot entered the final take-off zone at low speed.
    candidate_indices = list(range(1, len(gap_list)))
    rng.shuffle(candidate_indices)
    wanted_approaches = min(
        len(candidate_indices), rng.randint(*settings["approach_patch_count"])
    )
    approach_count = 0
    for index in candidate_indices:
        if approach_count >= wanted_approaches or len(patches) >= target_count:
            break
        gap_x, _gap_width = gap_list[index]
        previous_gap_end = gap_list[index - 1][0] + gap_list[index - 1][1]
        earliest_patch_x = max(min_x, previous_gap_end + GAP_EDGE_CLEARANCE_FOR_PATCH)
        max_available_width = int(gap_x - GAP_APPROACH_RUNUP - earliest_patch_x)
        if max_available_width < min_width:
            continue
        width = rng.randint(min_width, min(max_width, max_available_width))
        x = int(gap_x - GAP_APPROACH_RUNUP - width)
        kind = rng.choices(TERRAIN_TYPES, weights=weights, k=1)[0]
        if _patch_is_safe(x, width, gap_list, patches, start_x, goal_x):
            patches.append((x, width, kind))
            approach_count += 1

    # Fill remaining slots with terrain on available stretches of solid ground.
    # It is acceptable to generate fewer patches than requested if a course has
    # no remaining safe space; it is not acceptable to violate the run-up zone.
    attempts = 0
    while len(patches) < target_count and attempts < MAX_PLACEMENT_ATTEMPTS:
        attempts += 1
        if max_end - min_x < min_width:
            break
        width = rng.randint(min_width, max_width)
        x = rng.randint(min_x, max_end - width)
        if not _patch_is_safe(x, width, gap_list, patches, start_x, goal_x):
            continue
        kind = rng.choices(TERRAIN_TYPES, weights=weights, k=1)[0]
        patches.append((x, width, kind))

    return sorted(patches, key=lambda patch: patch[0])


def generate_surface_patches(
    gaps: Iterable[tuple[int, int]],
    difficulty: str = "easy",
    seed: int | None = None,
    start_x: int = 100,
    goal_x: int = 2400,
) -> list[tuple[int, int, str]]:
    """Generate surface patches as ``(x, width, kind)`` tuples.

    The function signature and return representation remain compatible with the
    original VoidRunner generator. The final acceleration runway before a gap
    is protected for every difficulty.
    """
    _check_inputs(difficulty, start_x, goal_x)
    effective_seed = _effective_seed(seed)
    return _generate_surface_patches(
        gaps, difficulty, random.Random(effective_seed), start_x, goal_x
    )


def _generate_obstacles(
    gaps: Iterable[tuple[int, int]],
    surface_patches: Iterable[tuple[int, int, str]],
    difficulty: str,
    rng: random.Random,
    start_x: int | float,
    goal_x: int | float,
) -> list[tuple[int, int, int]]:
    """Generate bounded rocks as ``(x, width, height)`` tuples.

    At most one rock is placed in each post-gap pocket. Selected rocks use a
    smaller gap-plus-rock profile; remaining rocks vary more in size but still
    leave the next gap's protected run-up clear. If a pocket cannot fit a rock
    safely, it is skipped rather than retried indefinitely.
    """
    settings = DIFFICULTY_SETTINGS[difficulty]
    gap_list = list(gaps)
    target_count = min(rng.randint(*settings["rock_count"]), len(gap_list))
    min_width, max_width = settings["rock_width"]
    min_height, max_height = settings["rock_height"]
    rocks: list[tuple[int, int, int]] = []
    min_x = int(start_x + SPAWN_CLEARANCE)
    max_x = int(goal_x - GOAL_CLEARANCE)

    combo_min, combo_max = settings["combo_rock_count"]
    combo_remaining = min(target_count, rng.randint(combo_min, combo_max), len(gap_list))
    candidate_indices = list(range(len(gap_list)))
    rng.shuffle(candidate_indices)
    combo_offset = settings["combo_rock_gap_offset"]
    combo_width = settings["combo_rock_width"]
    combo_height = settings["combo_rock_height"]

    for index in candidate_indices:
        if len(rocks) >= target_count:
            break

        gap_x, gap_width = gap_list[index]
        gap_end = gap_x + gap_width
        next_gap_x = gap_list[index + 1][0] if index + 1 < len(gap_list) else max_x + 1
        latest_rock_end = min(max_x, next_gap_x - GAP_APPROACH_RUNUP)
        available_after_edge = int(latest_rock_end - gap_end)
        if available_after_edge < 18 + min_width:
            continue

        use_combo = combo_remaining > 0
        if use_combo:
            width = rng.randint(*combo_width)
            height = rng.randint(*combo_height)
            offset_low, offset_high = combo_offset
            offset_low = max(18, offset_low)
            offset_high = min(offset_high, available_after_edge - width)
            if offset_high < offset_low:
                # This pocket cannot fit the intended combo; use a normal rock
                # if that rock can fit with its full clearances.
                use_combo = False

        if not use_combo:
            feasible_max_width = min(max_width, available_after_edge - 18)
            if feasible_max_width < min_width:
                continue
            width = rng.randint(min_width, feasible_max_width)
            height = rng.randint(min_height, max_height)
            offset_low = 18
            offset_high = min(70, available_after_edge - width)
            if offset_high < offset_low:
                continue

        offset = rng.randint(offset_low, offset_high)
        x = int(gap_end + offset)
        if x < min_x or x + width > latest_rock_end:
            continue
        if any(_overlaps(x, width, gx, gw, GAP_GEOMETRY_CLEARANCE_FOR_ROCK)
               for gx, gw in gap_list):
            continue
        if any(_overlaps(x, width, rx, rw, ROCK_SEPARATION) for rx, rw, _ in rocks):
            continue

        rocks.append((x, width, height))
        if use_combo:
            combo_remaining -= 1

    # If the last few pockets were too narrow, a final rock may be placed after
    # the last gap, provided the goal clearance and isolation constraints allow.
    if len(rocks) < target_count and gap_list:
        last_gap_end = gap_list[-1][0] + gap_list[-1][1]
        tail_start = max(min_x, last_gap_end + ISOLATED_ROCK_GAP_CLEARANCE)
        feasible_room = max_x - tail_start
        if feasible_room >= min_width:
            width = rng.randint(min_width, min(max_width, feasible_room))
            height = rng.randint(min_height, max_height)
            x = rng.randint(tail_start, max_x - width)
            if not any(_overlaps(x, width, gx, gw, ISOLATED_ROCK_GAP_CLEARANCE)
                       for gx, gw in gap_list) and not any(
                _overlaps(x, width, rx, rw, ROCK_SEPARATION) for rx, rw, _ in rocks
            ):
                rocks.append((x, width, height))

    return sorted(rocks, key=lambda rock: rock[0])

def generate_obstacles(
    gaps: Iterable[tuple[int, int]],
    surface_patches: Iterable[tuple[int, int, str]] | None = None,
    difficulty: str = "easy",
    seed: int | None = None,
    start_x: int = 100,
    goal_x: int = 2400,
) -> list[tuple[int, int, int]]:
    """Generate rock obstacles as ``(start_x, width, height)`` tuples."""
    _check_inputs(difficulty, start_x, goal_x)
    effective_seed = _effective_seed(seed)
    return _generate_obstacles(
        gaps, surface_patches or [], difficulty, random.Random(effective_seed), start_x, goal_x
    )


def generate_layout(
    difficulty: str = "easy",
    seed: int | None = None,
    start_x: int = 100,
    goal_x: int = 2400,
) -> dict[str, Any]:
    """Generate a complete, JSON-serializable terrain layout snapshot."""
    _check_inputs(difficulty, start_x, goal_x)
    effective_seed = _effective_seed(seed)
    gaps = generate_terrain(difficulty, effective_seed, start_x, goal_x)
    patches = generate_surface_patches(gaps, difficulty, effective_seed, start_x, goal_x)
    rocks = generate_obstacles(gaps, patches, difficulty, effective_seed, start_x, goal_x)

    layout: dict[str, Any] = {
        "generator_version": GENERATOR_VERSION,
        "seed": effective_seed,
        "difficulty": difficulty,
        "start_x": start_x,
        "goal_x": goal_x,
        "world_width": goal_x,
        "coordinate_unit": "pixels",
        "generator_settings": {
            name: list(values) for name, values in DIFFICULTY_SETTINGS[difficulty].items()
        },
        "generation_clearances": {
            "gap_approach_runup": GAP_APPROACH_RUNUP,
            "patch_gap_edge_clearance": GAP_EDGE_CLEARANCE_FOR_PATCH,
            "patch_separation": PATCH_SEPARATION,
            "minimum_gap_spacing": MIN_GAP_SPACING,
            "isolated_rock_gap_clearance": ISOLATED_ROCK_GAP_CLEARANCE,
            "rock_separation": ROCK_SEPARATION,
        },
        "feasibility_profile": {
            "version": FEASIBILITY_PROFILE_VERSION,
            "max_gap_width": MAX_GENERATED_GAP_WIDTH,
            "max_rock_width": MAX_GENERATED_ROCK_WIDTH,
            "max_rock_height": MAX_GENERATED_ROCK_HEIGHT,
            "gap_approach_runup": GAP_APPROACH_RUNUP,
            "min_gap_spacing": MIN_GAP_SPACING,
            "note": "physics-informed constraints; verify with actual env rollout",
        },
        "gaps": gaps,
        "surface_patches": patches,
        "rocks": rocks,
    }
    errors = validate_layout(layout, enforce_feasibility=True)
    if errors:
        raise RuntimeError("Generated layout failed validation: " + "; ".join(errors))
    return layout


def validate_layout(
    layout: dict[str, Any],
    *,
    enforce_feasibility: bool | None = None,
) -> list[str]:
    """Check geometric consistency and optionally new traversal constraints.

    Old snapshots remain structurally loadable. New generated layouts are
    always validated against the stricter feasibility profile.
    """
    errors: list[str] = []
    try:
        start_x = float(layout["start_x"])
        goal_x = float(layout["goal_x"])
        gaps = [(int(x), int(w)) for x, w in layout["gaps"]]
        patches = [(int(x), int(w), str(k)) for x, w, k in layout["surface_patches"]]
        rocks = [(int(x), int(w), int(h)) for x, w, h in layout["rocks"]]
    except (KeyError, TypeError, ValueError) as exc:
        return [f"Malformed layout data: {exc}"]

    if enforce_feasibility is None:
        enforce_feasibility = int(layout.get("generator_version", 0)) >= FEASIBILITY_PROFILE_VERSION

    if goal_x <= start_x:
        errors.append("goal_x must be greater than start_x")
    if gaps != sorted(gaps, key=lambda gap: gap[0]):
        errors.append("gaps are not sorted by x")
    if patches != sorted(patches, key=lambda patch: patch[0]):
        errors.append("surface patches are not sorted by x")
    if rocks != sorted(rocks, key=lambda rock: rock[0]):
        errors.append("rocks are not sorted by x")

    for x, w in gaps:
        if w <= 0:
            errors.append(f"gap at {x} has non-positive width")
        if x < start_x + SPAWN_CLEARANCE:
            errors.append(f"gap at {x} intrudes into the protected spawn region")
        if x + w > goal_x - GOAL_CLEARANCE:
            errors.append(f"gap at {x} intrudes into the protected goal region")
        if enforce_feasibility and w > MAX_GENERATED_GAP_WIDTH:
            errors.append(f"gap at {x} is wider than the current feasible cap ({MAX_GENERATED_GAP_WIDTH}px)")
    for (x1, w1), (x2, w2) in zip(gaps, gaps[1:]):
        spacing = x2 - (x1 + w1)
        if _overlaps(x1, w1, x2, w2):
            errors.append(f"gaps at {x1} and {x2} overlap")
        if enforce_feasibility and spacing < MIN_GAP_SPACING:
            errors.append(
                f"only {spacing}px of solid ground between gaps at {x1} and {x2}; "
                f"at least {MIN_GAP_SPACING}px is required by the current jump physics"
            )

    for x, w, kind in patches:
        if w <= 0:
            errors.append(f"surface patch at {x} has non-positive width")
        if x < start_x + SPAWN_CLEARANCE or x + w > goal_x - GOAL_CLEARANCE:
            errors.append(f"surface patch at {x} intrudes into a protected start/goal region")
        if kind not in TERRAIN_TYPES:
            errors.append(f"surface patch at {x} has unknown type {kind!r}")
        for gx, gw in gaps:
            if _overlaps(x, w, gx, gw, GAP_EDGE_CLEARANCE_FOR_PATCH):
                errors.append(f"surface patch at {x} overlaps or is too close to a gap at {gx}")
                break
            if enforce_feasibility and x < gx and x + w > gx - GAP_APPROACH_RUNUP:
                errors.append(
                    f"surface patch at {x} enters the {GAP_APPROACH_RUNUP}px acceleration runway before gap {gx}"
                )
                break
    for (x1, w1, _), (x2, w2, _) in zip(patches, patches[1:]):
        if _overlaps(x1, w1, x2, w2, PATCH_SEPARATION):
            errors.append(f"surface patches at {x1} and {x2} are too close")

    for x, w, h in rocks:
        if w <= 0 or h <= 0:
            errors.append(f"rock at {x} has non-positive dimensions")
        if x < start_x + SPAWN_CLEARANCE or x + w > goal_x - GOAL_CLEARANCE:
            errors.append(f"rock at {x} intrudes into a protected start/goal region")
        if any(_overlaps(x, w, gx, gw, GAP_GEOMETRY_CLEARANCE_FOR_ROCK) for gx, gw in gaps):
            errors.append(f"rock at {x} overlaps or geometrically intrudes into a gap")
        if enforce_feasibility:
            if w > MAX_GENERATED_ROCK_WIDTH:
                errors.append(f"rock at {x} is wider than the current cap ({MAX_GENERATED_ROCK_WIDTH}px)")
            if h > MAX_GENERATED_ROCK_HEIGHT:
                errors.append(f"rock at {x} is taller than the current cap ({MAX_GENERATED_ROCK_HEIGHT}px)")
            for gx, _gw in gaps:
                if x < gx and x + w > gx - GAP_APPROACH_RUNUP:
                    errors.append(
                        f"rock at {x} intrudes into the {GAP_APPROACH_RUNUP}px acceleration runway before gap {gx}"
                    )
                    break
    for (x1, w1, _), (x2, w2, _) in zip(rocks, rocks[1:]):
        if _overlaps(x1, w1, x2, w2, ROCK_SEPARATION):
            errors.append(f"rocks at {x1} and {x2} are too close")

    return errors


def save_layout(layout: dict[str, Any], path: str | Path) -> None:
    """Save an actual terrain layout to JSON for later restoration."""
    errors = validate_layout(layout)
    if errors:
        raise ValueError("Cannot save invalid layout: " + "; ".join(errors))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(layout, file, indent=2, sort_keys=True)
        file.write("\n")


def load_layout(path: str | Path) -> dict[str, Any]:
    """Load a stored layout; v5 snapshots also receive feasibility validation."""
    with Path(path).open("r", encoding="utf-8") as file:
        layout = json.load(file)
    if not isinstance(layout, dict):
        raise ValueError("Layout JSON must contain an object")
    required = {
        "generator_version", "seed", "difficulty", "start_x", "goal_x",
        "world_width", "coordinate_unit", "generator_settings",
        "gaps", "surface_patches", "rocks",
    }
    missing = sorted(required - layout.keys())
    if missing:
        raise ValueError(f"Layout snapshot is missing fields: {', '.join(missing)}")
    errors = validate_layout(layout)
    if errors:
        raise ValueError("Loaded layout is invalid: " + "; ".join(errors))
    layout["gaps"] = [(int(x), int(w)) for x, w in layout["gaps"]]
    layout["surface_patches"] = [
        (int(x), int(w), str(kind)) for x, w, kind in layout["surface_patches"]
    ]
    layout["rocks"] = [(int(x), int(w), int(h)) for x, w, h in layout["rocks"]]
    return layout


def print_layout_report(layout: dict[str, Any]) -> None:
    errors = validate_layout(layout, enforce_feasibility=True)
    print(
        f"{layout['difficulty'].upper()} | seed={layout['seed']} | "
        f"gaps={len(layout['gaps'])}, patches={len(layout['surface_patches'])}, "
        f"rocks={len(layout['rocks'])}"
    )
    print("  Gaps:", ", ".join(f"{x}-{x + w} (w={w})" for x, w in layout["gaps"]) or "none")
    print("  Patches:", ", ".join(
        f"{kind}@{x}-{x + w}" for x, w, kind in layout["surface_patches"]
    ) or "none")
    print("  Rocks:", ", ".join(
        f"x={x}, w={w}, h={h}" for x, w, h in layout["rocks"]
    ) or "none")
    if errors:
        print("  FEASIBILITY ERRORS:")
        for error in errors:
            print("   -", error)
    else:
        print(
            f"  Feasibility profile: PASS (run-up={GAP_APPROACH_RUNUP}px, "
            f"min gap spacing={MIN_GAP_SPACING}px, max gap={MAX_GENERATED_GAP_WIDTH}px, "
            f"max rock={MAX_GENERATED_ROCK_WIDTH}x{MAX_GENERATED_ROCK_HEIGHT}px)"
        )
        print("  Note: actual traversal still depends on the robot's decisions and physics rollout.")


def _demo() -> None:
    for difficulty in ("easy", "medium", "hard"):
        print(f"\n{'=' * 18} {difficulty.upper()} {'=' * 18}")
        for seed in (1, 2, 11):
            print_layout_report(generate_layout(difficulty, seed=seed))

    sample = generate_layout("hard", seed=42)
    snapshot = Path("results") / "terrain_generator_smoke_test.json"
    save_layout(sample, snapshot)
    restored = load_layout(snapshot)
    assert restored == sample, "Saved layout did not round-trip exactly"
    print(f"\nSnapshot round-trip: PASS ({snapshot})")


if __name__ == "__main__":
    _demo()
