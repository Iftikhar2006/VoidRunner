"""Run the trained VoidRunner DQN in a visual episode.

From the repository root:
    python demo.py
    python demo.py --difficulty hard --seed 10000
    python demo.py --difficulty medium --seed 123 --no-render
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from agent.dqn import DQNAgent
from environment.terrain_env import ACTION_NAMES, ENVIRONMENT_VERSION, TerrainEnv

DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "voidrunner_fullterrain_best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch the trained VoidRunner DQN navigate a generated course."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help=f"trained checkpoint (default: {DEFAULT_CHECKPOINT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="easy")
    parser.add_argument("--seed", type=int, default=10000, help="terrain seed; repeated runs are reproducible")
    parser.add_argument("--episodes", type=int, default=1, help="number of sequential episodes")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-render", action="store_true", help="run headlessly and print progress")
    parser.add_argument("--log-every", type=int, default=50, help="print one progress line every N steps")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but PyTorch cannot see a CUDA device. Try --device cpu.")
    return requested


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be at least 1")
    if args.log_every <= 0:
        raise SystemExit("--log-every must be positive")
    if not checkpoint.is_file():
        raise SystemExit(
            f"Trained checkpoint not found: {checkpoint}\n"
            "Copy checkpoints/voidrunner_multigap_transfer_seed46/best.pt to "
            "models/voidrunner_fullterrain_best.pt, or pass --checkpoint PATH."
        )

    device = resolve_device(args.device)
    render_mode = None if args.no_render else "human"
    env = TerrainEnv(difficulty=args.difficulty, render_mode=render_mode, max_steps=args.max_steps)
    try:
        agent = DQNAgent.from_env(env, device=device, seed=args.seed)
        metadata = agent.load_checkpoint(
            checkpoint,
            expected_environment_version=ENVIRONMENT_VERSION,
            load_optimizer=False,
        )
        agent.online_network.eval()

        print("VOIDRUNNER — TRAINED DQN DEMO")
        print("=" * 60)
        print(f"Checkpoint: {checkpoint}")
        print(f"Checkpoint episode: {metadata.get('episode', 'unknown')}")
        print(f"Environment: {ENVIRONMENT_VERSION}")
        print(f"Difficulty: {args.difficulty}; device: {device}; render: {not args.no_render}")
        print("Policy: greedy DQN (epsilon = 0); no rule-based controller is used.")
        print("Close the game window or press Esc to stop the visual demo.\n")

        pygame: Any = None
        if not args.no_render:
            try:
                import pygame as pygame_module
                pygame = pygame_module
            except ImportError as exc:
                raise SystemExit("Pygame is required for the visual demo. Install it with pip install pygame.") from exc

        for episode_index in range(args.episodes):
            episode_seed = args.seed + episode_index
            observation, info = env.reset(seed=episode_seed)
            total_reward = 0.0
            user_quit = False
            print(f"Episode {episode_index + 1}/{args.episodes} | seed={episode_seed} | starting...")

            while True:
                action = agent.select_action(observation, epsilon=0.0)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)

                if not args.no_render:
                    env.render()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (
                            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                        ):
                            user_quit = True
                            break
                    if user_quit:
                        print("Demo stopped by user.")
                        break

                if info["steps"] % args.log_every == 0:
                    print(
                        f"  step={info['steps']:4d} x={info['x']:7.1f} "
                        f"speed={info['speed']:5.2f} energy={info['energy_remaining']:6.2f} "
                        f"action={ACTION_NAMES[action]}"
                    )

                if terminated or truncated:
                    outcome = "SUCCESS" if info.get("success", False) else f"FAIL: {info.get('failure_reason') or 'unknown'}"
                    print(
                        f"  {outcome} | steps={info['steps']} | "
                        f"energy_used={info.get('energy_consumed', 0.0):.2f} | "
                        f"return={total_reward:.3f}\n"
                    )
                    break
                if user_quit:
                    break

            if user_quit:
                break
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
