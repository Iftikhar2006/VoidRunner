
import argparse
from pathlib import Path

import numpy as np
import pygame
import torch

from agent.dqn import DQN
from environment.terrain_env import TerrainEnv


def evaluate_dqn(episodes=10, max_steps=400, render=False):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = Path("checkpoints/voidrunner_dqn.pt")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = DQN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Evaluation device:", device)
    print("Checkpoint episode:", checkpoint["episode"])
    print("Exploration: disabled")
    print("Rendering:", render)

    try:
        for difficulty in ["easy", "medium", "hard"]:
            env = TerrainEnv(
                render_mode="human" if render else None,
                difficulty=difficulty,
            )

            rewards = []
            steps_list = []
            successes = 0
            falls = 0
            timeouts = 0

            try:
                for episode in range(episodes):
                    seed = 1000 + episode
                    state, _ = env.reset(seed=seed)

                    if difficulty == "medium":
                        print(
                            "Generated gaps (start, width):",
                            env.gaps,
                        )

                    episode_reward = 0.0
                    steps_taken = 0
                    success = False
                    fell = False
                    actions_taken = []
                    position_history = []

                    for step in range(max_steps):
                        if render:
                            env.render()

                        state_tensor = torch.as_tensor(
                            state,
                            dtype=torch.float32,
                            device=device,
                        ).unsqueeze(0)

                        with torch.no_grad():
                            q_values = model(state_tensor)
                            action = int(
                                q_values.argmax(dim=1).item()
                            )

                        if (
                            difficulty == "medium"
                            and step >= max_steps - 20
                        ):
                            print(
                                f"Step={step + 1}, "
                                f"x={env.player_x:.1f}, "
                                f"y={env.player_y:.1f}, "
                                f"state={np.round(state, 3)}, "
                                f"Q-values="
                                f"{np.round(q_values.cpu().numpy()[0], 3)}, "
                                f"action={action}"
                            )

                        actions_taken.append(action)

                        (
                            next_state,
                            reward,
                            terminated,
                            truncated,
                            info,
                        ) = env.step(action)

                        steps_taken += 1

                        position_history.append(
                            (
                                steps_taken,
                                round(env.player_x, 1),
                                action,
                            )
                        )

                        state = next_state
                        episode_reward += reward

                        if terminated or truncated:
                            success = info.get("success", False)
                            fell = (
                                info.get("reason") == "fell_into_gap"
                            )
                            break

                    rewards.append(episode_reward)
                    steps_list.append(steps_taken)

                    if success:
                        successes += 1
                    elif fell:
                        falls += 1
                    else:
                        timeouts += 1

                    print(
                        f"{difficulty:6s} | "
                        f"Episode {episode + 1:2d}/{episodes} | "
                        f"Seed: {seed} | "
                        f"Reward: {episode_reward:8.2f} | "
                        f"Steps: {steps_taken:3d} | "
                        f"Success: {success}"
                    )

                    print(
                        "Last 30 actions:",
                        actions_taken[-30:],
                    )

                    print("Last 30 positions (step, x, action):")
                    for entry in position_history[-30:]:
                        print(entry)

                    print(
                        "Final player position:",
                        round(env.player_x, 1),
                    )

            finally:
                env.close()

            print("\n" + "=" * 50)
            print(f"Difficulty: {difficulty.upper()}")
            print(f"Successes: {successes}/{episodes}")
            print(
                f"Success rate: "
                f"{100 * successes / episodes:.1f}%"
            )
            print(f"Falls: {falls}")
            print(f"Timeouts: {timeouts}")
            print(f"Average reward: {np.mean(rewards):.2f}")
            print(f"Average steps: {np.mean(steps_list):.1f}")
            print("=" * 50 + "\n")

    finally:
        if render:
            pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()

    if args.episodes < 1 or args.max_steps < 1:
        parser.error("--episodes and --max-steps must be positive")

    evaluate_dqn(
        episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
    )
