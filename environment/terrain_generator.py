"""Deterministic terrain generation for VoidRunner.

Public interfaces preserved:
    generate_terrain(difficulty, seed, start_x, goal_x) -> [(x, width), ...]
    generate_surface_patches(gaps, difficulty, seed, start_x, goal_x)
        -> [(x, width, terrain_type), ...]

New helpers:
    generate_obstacles(...)
    generate_layout(...)
    validate_layout(layout)
    save_layout(layout, path)
    load_layout(path)

Coordinates are world-space pixels. This module checks layout geometry, not
whether a particular physics model can successfully traverse every layout.
That requires tests using terrain_env.py's actual jump and collision physics.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable


GENERATOR_VERSION = 3

# These are generation parameters, not claims that every generated course is
# physically solvable. Difficulty will be verified against the real environment.
DIFFICULTY_SETTINGS: dict[str, dict[str, tuple[int, ...]]] = {
    "easy": {
        "gap_width": (40, 60),
        "gap_spacing": (235, 320),
        "gap_count": (5, 7),
        "patch_count": (7, 10),
        "patch_width": (45, 85),
        "rock_count": (3, 5),
        "rock_width": (24, 42),
        "rock_height": (18, 34),
        "terrain_weights": (5, 4, 1),
    },
    "medium": {
        "gap_width": (50, 78),
        "gap_spacing": (165, 230),
        "gap_count": (6, 8),
        "patch_count": (8, 11),
        "patch_width": (50, 90),
        "rock_count": (4, 6),
        "rock_width": (28, 48),
        "rock_height": (22, 42),
        "terrain_weights": (3, 4, 3),
    },
    "hard": {
        "gap_width": (62, 95),
        "gap_spacing": (125, 175),
        "gap_count": (7, 9),
        "patch_count": (8, 11),
        "patch_width": (60, 100),
        "rock_count": (6, 9),
        "rock_width": (30, 55),
        "rock_height": (26, 48),
        "terrain_weights": (2, 3, 5),
    },
}

TERRAIN_TYPES = ("rubble", "mud", "hazard")

# Clearances are in world-space pixels.
SPAWN_CLEARANCE = 220
GOAL_CLEARANCE = 100
GAP_CLEARANCE_FOR_PATCH = 12
PATCH_SEPARATION = 25
# Placement uses conservative recovery buffers so the physics does not routinely
# generate a rock immediately after a gap or another rock. The smaller geometry
# clearances below remain useful for validating older, explicitly saved layouts.
GAP_CLEARANCE_FOR_ROCK = 180  # generation buffer around gaps
ROCK_SEPARATION = 105  # generation buffer between rocks
GAP_GEOMETRY_CLEARANCE_FOR_ROCK = 18
ROCK_GEOMETRY_SEPARATION = 24
MAX_PLACEMENT_ATTEMPTS = 3000


def _check_inputs(difficulty: str, start_x: int, goal_x: int) -> None:
    if difficulty not in DIFFICULTY_SETTINGS:
        allowed = ", ".join(DIFFICULTY_SETTINGS)
        raise ValueError(f"difficulty must be one of: {allowed}")
    if not isinstance(start_x, (int, float)) or not isinstance(goal_x, (int, float)):
        raise TypeError("start_x and goal_x must be numbers")
    if goal_x <= start_x + SPAWN_CLEARANCE + GOAL_CLEARANCE:
        raise ValueError("goal_x must leave enough room for a route")


def _effective_seed(seed: int | None) -> int:
    """Resolve None to an explicit seed so a generated layout can be replayed."""
    if seed is None:
        return random.SystemRandom().randrange(0, 2**32)
    if not isinstance(seed, int):
        raise TypeError("seed must be an int or None")
    return seed


def _overlaps(a_start: float, a_width: float, b_start: float, b_width: float,
              clearance: float = 0) -> bool:
    """Return True if two intervals overlap or are closer than clearance."""
    return (a_start < b_start + b_width + clearance
            and a_start + a_width + clearance > b_start)


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
    """Generate gaps as ``(start_x, width)`` tuples.

    This is the original VoidRunner public interface. Supplying the same seed
    and parameters produces the same gaps.
    """
    _check_inputs(difficulty, start_x, goal_x)
    effective_seed = _effective_seed(seed)
    return _generate_terrain(difficulty, random.Random(effective_seed), start_x, goal_x)


def _generate_surface_patches(
    gaps: Iterable[tuple[int, int]],
    difficulty: str,
    rng: random.Random,
    start_x: int | float,
    goal_x: int | float,
) -> list[tuple[int, int, str]]:
    settings = DIFFICULTY_SETTINGS[difficulty]
    patch_count = rng.randint(*settings["patch_count"])
    min_width, max_width = settings["patch_width"]
    patches: list[tuple[int, int, str]] = []
    gap_list = list(gaps)

    min_x = int(start_x + SPAWN_CLEARANCE)
    max_start = int(goal_x - GOAL_CLEARANCE)
    if max_start <= min_x:
        return patches

    attempts = 0
    while len(patches) < patch_count and attempts < MAX_PLACEMENT_ATTEMPTS:
        attempts += 1
        width = rng.randint(min_width, max_width)
        if max_start - width < min_x:
            break
        x = rng.randint(min_x, max_start - width)
        kind = rng.choices(
            TERRAIN_TYPES, weights=settings["terrain_weights"], k=1
        )[0]

        if any(_overlaps(x, width, gx, gw, GAP_CLEARANCE_FOR_PATCH)
               for gx, gw in gap_list):
            continue
        if any(_overlaps(x, width, old_x, old_width, PATCH_SEPARATION)
               for old_x, old_width, _ in patches):
            continue

        patches.append((x, width, kind))

    return sorted(patches, key=lambda patch: patch[0])


def generate_surface_patches(
    gaps: Iterable[tuple[int, int]],
    difficulty: str = "easy",
    seed: int | None = None,
    start_x: int = 100,
    goal_x: int = 2400,
) -> list[tuple[int, int, str]]:
    """Generate visual/mechanical surface patches as ``(x, width, kind)``.

    Existing interface retained. A patch is kept clear of gaps and other
    patches. Whether a patch changes physics is the environment's responsibility.
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
    """Generate rocks as ``(x, width, height)`` tuples on solid ground."""
    settings = DIFFICULTY_SETTINGS[difficulty]
    target_count = rng.randint(*settings["rock_count"])
    min_width, max_width = settings["rock_width"]
    min_height, max_height = settings["rock_height"]
    gap_list = list(gaps)
    # Surface patches describe the material of the ground, so a rock may sit
    # on top of rubble/mud without this being an accidental overlap.
    rocks: list[tuple[int, int, int]] = []

    min_x = int(start_x + SPAWN_CLEARANCE)
    max_start = int(goal_x - GOAL_CLEARANCE)
    attempts = 0

    while len(rocks) < target_count and attempts < MAX_PLACEMENT_ATTEMPTS:
        attempts += 1
        width = rng.randint(min_width, max_width)
        height = rng.randint(min_height, max_height)
        if max_start - width < min_x:
            break
        x = rng.randint(min_x, max_start - width)

        if any(_overlaps(x, width, gx, gw, GAP_CLEARANCE_FOR_ROCK)
               for gx, gw in gap_list):
            continue
        if any(_overlaps(x, width, rx, rw, ROCK_SEPARATION)
               for rx, rw, _ in rocks):
            continue

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
    """Generate rock obstacles as ``(start_x, width, height)`` tuples.

    Rocks are placed on solid ground, clear of gaps and other rocks. They may
    appear on a terrain patch because a patch describes ground material. This
    helper adds data for the next environment phase; it does not yet add physics.
    """
    _check_inputs(difficulty, start_x, goal_x)
    effective_seed = _effective_seed(seed)
    return _generate_obstacles(
        gaps,
        surface_patches or [],
        difficulty,
        random.Random(effective_seed),
        start_x,
        goal_x,
    )


def generate_layout(
    difficulty: str = "easy",
    seed: int | None = None,
    start_x: int = 100,
    goal_x: int = 2400,
) -> dict[str, Any]:
    """Generate a complete, JSON-serializable terrain layout snapshot.

    The returned dictionary stores actual geometry as well as the seed. Loading
    a saved snapshot therefore does not rely solely on the generator producing
    exactly the same output after future code changes.
    """
    _check_inputs(difficulty, start_x, goal_x)
    effective_seed = _effective_seed(seed)
    # Use the same seed semantics as the individual public functions. This makes
    # layout["gaps"] match generate_terrain(..., seed=seed), and so on.
    gaps = generate_terrain(difficulty, effective_seed, start_x, goal_x)
    patches = generate_surface_patches(
        gaps, difficulty, effective_seed, start_x, goal_x
    )
    rocks = generate_obstacles(
        gaps, patches, difficulty, effective_seed, start_x, goal_x
    )

    layout: dict[str, Any] = {
        "generator_version": GENERATOR_VERSION,
        "seed": effective_seed,
        "difficulty": difficulty,
        "start_x": start_x,
        "goal_x": goal_x,
        "world_width": goal_x,
        "coordinate_unit": "pixels",
        "generator_settings": {
            name: list(values)
            for name, values in DIFFICULTY_SETTINGS[difficulty].items()
        },
        "generation_clearances": {
            "patch_gap_clearance": GAP_CLEARANCE_FOR_PATCH,
            "patch_separation": PATCH_SEPARATION,
            "rock_gap_recovery_buffer": GAP_CLEARANCE_FOR_ROCK,
            "rock_separation_buffer": ROCK_SEPARATION,
        },
        "gaps": gaps,
        "surface_patches": patches,
        "rocks": rocks,
    }
    errors = validate_layout(layout)
    if errors:
        raise RuntimeError("Generated invalid layout: " + "; ".join(errors))
    return layout


def validate_layout(layout: dict[str, Any]) -> list[str]:
    """Check geometric consistency; an empty list means no geometry errors.

    This does not establish jump reachability or full route solvability.
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
    for (x1, w1), (x2, w2) in zip(gaps, gaps[1:]):
        if _overlaps(x1, w1, x2, w2):
            errors.append(f"gaps at {x1} and {x2} overlap")

    for x, w, kind in patches:
        if w <= 0:
            errors.append(f"surface patch at {x} has non-positive width")
        if x < start_x + SPAWN_CLEARANCE or x + w > goal_x - GOAL_CLEARANCE:
            errors.append(f"surface patch at {x} intrudes into a protected start/goal region")
        if kind not in TERRAIN_TYPES:
            errors.append(f"surface patch at {x} has unknown type {kind!r}")
        if any(_overlaps(x, w, gx, gw, GAP_CLEARANCE_FOR_PATCH) for gx, gw in gaps):
            errors.append(f"surface patch at {x} overlaps or is too close to a gap")
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
    for (x1, w1, _), (x2, w2, _) in zip(rocks, rocks[1:]):
        if _overlaps(x1, w1, x2, w2, ROCK_GEOMETRY_SEPARATION):
            errors.append(f"rocks at {x1} and {x2} overlap or are geometrically too close")

    return errors


def save_layout(layout: dict[str, Any], path: str | Path) -> None:
    """Save an actual terrain layout to JSON for exact later restoration."""
    errors = validate_layout(layout)
    if errors:
        raise ValueError("Cannot save invalid layout: " + "; ".join(errors))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(layout, file, indent=2, sort_keys=True)
        file.write("\n")


def load_layout(path: str | Path) -> dict[str, Any]:
    """Load a saved terrain layout, preserving its recorded geometry."""
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

    # Normalize JSON arrays back to the tuple shapes used by the generator API.
    layout["gaps"] = [(int(x), int(w)) for x, w in layout["gaps"]]
    layout["surface_patches"] = [
        (int(x), int(w), str(kind)) for x, w, kind in layout["surface_patches"]
    ]
    layout["rocks"] = [(int(x), int(w), int(h)) for x, w, h in layout["rocks"]]
    return layout


def print_layout_report(layout: dict[str, Any]) -> None:
    """Print a human-readable geometry report for inspecting generated maps."""
    errors = validate_layout(layout)
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
        print("  GEOMETRY ERRORS:")
        for error in errors:
            print("   -", error)
    else:
        print("  Geometry audit: PASS (physical jump feasibility not yet tested)")


def _demo() -> None:
    """Diagnostic run: inspect multiple fixed seeds at every difficulty."""
    for difficulty in ("easy", "medium", "hard"):
        print(f"\n{'=' * 18} {difficulty.upper()} {'=' * 18}")
        for seed in (1, 2, 11):
            layout = generate_layout(difficulty=difficulty, seed=seed)
            print_layout_report(layout)

    # Snapshot round-trip test using a fixed seed.
    sample = generate_layout("medium", seed=42)
    snapshot = Path("results") / "terrain_generator_smoke_test.json"
    save_layout(sample, snapshot)
    restored = load_layout(snapshot)
    assert restored == sample, "Saved layout did not round-trip exactly"
    print(f"\nSnapshot round-trip: PASS ({snapshot})")


if __name__ == "__main__":
    _demo()
