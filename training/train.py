import argparse
import csv
import random
from pathlib import Path

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

from agent.dqn import DQN
from agent.replay_buffer import ReplayBuffer
from environment.terrain_env import TerrainEnv


def select_difficulty(episode, total_episodes=None):
    if episode <= 250:
        return "easy"
    elif episode <= 375:
        return "medium"
    return "hard"


def select_action(model, state, epsilon, device):
    if random.random() < epsilon:
        return random.randrange(6)

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state_tensor)

    return int(q_values.argmax(dim=1).item())


def save_checkpoint(path, episode, model, optimizer, total_steps):
    torch.save(
        {
            "episode": episode,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "total_steps": total_steps,
        },
        path,
    )


def train_dqn(
    episodes=500,
    max_steps=600,
    batch_size=64,
    buffer_capacity=100_000,
    learning_rate=0.0005,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=50_000,
    min_buffer_size=1000,
    target_update_frequency=500,
    resume=False,
    render=False,
):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Training device:", device)

    results_dir = Path("results")
    checkpoints_dir = Path("checkpoints")

    results_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)

    log_path = results_dir / "training_log.csv"
    checkpoint_path = checkpoints_dir / "voidrunner_dqn.pt"

    env = TerrainEnv(
        render_mode="human" if render else None,
        difficulty="easy",
    )

    online_network = DQN().to(device)
    target_network = DQN().to(device)
    target_network.load_state_dict(online_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(
        online_network.parameters(),
        lr=learning_rate,
    )
    loss_function = nn.SmoothL1Loss()

    replay_buffer = ReplayBuffer(buffer_capacity)

    start_episode = 1
    total_steps = 0

    if resume:
        if not checkpoint_path.exists():
            env.close()
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        online_network.load_state_dict(
            checkpoint["model_state_dict"]
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_episode = checkpoint["episode"] + 1
        total_steps = checkpoint["total_steps"]

        target_network.load_state_dict(
            online_network.state_dict()
        )

        print("Resuming from episode:", checkpoint["episode"])
        print("Previous environment steps:", total_steps)

    if start_episode > episodes:
        env.close()
        print(
            f"Training already reached episode {start_episode - 1}. "
            f"Set --episodes to a number greater than "
            f"{start_episode - 1} to continue."
        )
        return

    episode_rewards = []
    episode_successes = []

    log_mode = "a" if resume and log_path.exists() else "w"

    try:
        with log_path.open(log_mode, newline="") as log_file:
            writer = csv.writer(log_file)

            if log_mode == "w" or log_path.stat().st_size == 0:
                writer.writerow(
                    [
                        "episode",
                        "difficulty",
                        "reward",
                        "steps",
                        "epsilon",
                        "success",
                    ]
                )

            for episode in range(start_episode, episodes + 1):
                difficulty = select_difficulty(episode)
                env.difficulty = difficulty

                state, _ = env.reset()

                episode_reward = 0.0
                success = False
                steps_taken = 0
                epsilon = epsilon_end

                for step in range(max_steps):
                    if render:
                        env.render()

                        if env.window is None:
                            raise KeyboardInterrupt

                    epsilon = max(
                        epsilon_end,
                        epsilon_start
                        - (epsilon_start - epsilon_end)
                        * total_steps
                        / epsilon_decay_steps,
                    )

                    action = select_action(
                        online_network,
                        state,
                        epsilon,
                        device,
                    )

                    (
                        next_state,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = env.step(action)

                    steps_taken += 1
                    total_steps += 1
                    episode_reward += reward

                    episode_done = terminated or truncated

                    if step == max_steps - 1 and not episode_done:
                        episode_done = True

                    replay_buffer.push(
                        state,
                        action,
                        reward,
                        next_state,
                        episode_done,
                    )

                    state = next_state

                    if len(replay_buffer) >= min_buffer_size:
                        (
                            states,
                            actions,
                            rewards,
                            next_states,
                            dones,
                        ) = replay_buffer.sample(batch_size)

                        states = torch.as_tensor(
                            states,
                            dtype=torch.float32,
                            device=device,
                        )
                        actions = torch.as_tensor(
                            actions,
                            dtype=torch.long,
                            device=device,
                        ).unsqueeze(1)
                        rewards = torch.as_tensor(
                            rewards,
                            dtype=torch.float32,
                            device=device,
                        )
                        next_states = torch.as_tensor(
                            next_states,
                            dtype=torch.float32,
                            device=device,
                        )
                        dones = torch.as_tensor(
                            dones,
                            dtype=torch.float32,
                            device=device,
                        )

                        current_q_values = online_network(
                            states
                        ).gather(1, actions).squeeze(1)

                        with torch.no_grad():
                            next_q_values = target_network(
                                next_states
                            ).max(dim=1).values

                            target_q_values = rewards + gamma * (
                                1.0 - dones
                            ) * next_q_values

                        loss = loss_function(
                            current_q_values,
                            target_q_values,
                        )

                        optimizer.zero_grad()
                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(
                            online_network.parameters(),
                            max_norm=10.0,
                        )

                        optimizer.step()

                        if total_steps % target_update_frequency == 0:
                            target_network.load_state_dict(
                                online_network.state_dict()
                            )

                    if episode_done:
                        success = info.get("success", False)
                        break

                episode_rewards.append(episode_reward)
                episode_successes.append(success)

                writer.writerow(
                    [
                        episode,
                        difficulty,
                        round(episode_reward, 3),
                        steps_taken,
                        round(epsilon, 4),
                        success,
                    ]
                )
                log_file.flush()

                if episode % 10 == 0 or episode == start_episode:
                    average_reward = np.mean(episode_rewards[-10:])
                    success_rate = (
                        np.mean(episode_successes[-10:]) * 100
                    )

                    print(
                        f"Episode {episode:4d}/{episodes} | "
                        f"Difficulty: {difficulty:6s} | "
                        f"Reward: {episode_reward:8.2f} | "
                        f"Avg reward (last 10): "
                        f"{average_reward:8.2f} | "
                        f"Success rate (last 10): "
                        f"{success_rate:5.1f}% | "
                        f"Epsilon: {epsilon:.3f}"
                    )

                if episode % 50 == 0:
                    save_checkpoint(
                        checkpoint_path,
                        episode,
                        online_network,
                        optimizer,
                        total_steps,
                    )

                if render:
                    env.render()

    finally:
        env.close()
        if render:
            pygame.quit()

    save_checkpoint(
        checkpoint_path,
        episode,
        online_network,
        optimizer,
        total_steps,
    )

    print("\nTraining finished.")
    print("Model saved to:", checkpoint_path)
    print("Training log saved to:", log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()

    train_dqn(
        episodes=args.episodes,
        max_steps=args.max_steps,
        resume=args.resume,
        render=args.render,
    )
Add this in that file
