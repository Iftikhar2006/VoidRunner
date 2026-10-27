"""Seedable replay buffer for VoidRunner DQN.

A transition stores terminated and truncated separately. Older callers may
still pass a fifth positional `done` argument; it is interpreted as a true
terminal for backward compatibility. New code should pass both keywords.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        capacity: int = 50_000,
        state_size: int = 18,
        action_size: int = 6,
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if state_size <= 0 or action_size <= 0:
            raise ValueError("state_size and action_size must be positive")
        self.capacity = int(capacity)
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool, bool]] = deque(
            maxlen=self.capacity
        )
        self.rng = np.random.default_rng(seed)

    def push(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool | None = None,
        *,
        terminated: bool | None = None,
        truncated: bool = False,
    ) -> None:
        """Add a transition.

        Use either old-style `push(..., done)` or new-style
        `push(..., terminated=..., truncated=...)`. For a time-limit
        truncation, store terminated=False and truncated=True, allowing DQN
        targets to bootstrap from the final observation.
        """
        if terminated is None:
            if done is None:
                raise TypeError("Provide either done or terminated")
            terminated = bool(done)
        elif done is not None:
            raise TypeError("Do not combine positional done with terminated; use terminated/truncated")

        state_array = np.asarray(state, dtype=np.float32)
        next_state_array = np.asarray(next_state, dtype=np.float32)
        if state_array.shape != (self.state_size,):
            raise ValueError(f"state must have shape ({self.state_size},), got {state_array.shape}")
        if next_state_array.shape != (self.state_size,):
            raise ValueError(
                f"next_state must have shape ({self.state_size},), got {next_state_array.shape}"
            )
        if not np.isfinite(state_array).all() or not np.isfinite(next_state_array).all():
            raise ValueError("state and next_state must contain only finite values")
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < self.action_size:
            raise ValueError(f"action must be an integer in [0, {self.action_size - 1}]")
        if not np.isfinite(float(reward)):
            raise ValueError("reward must be finite")

        # Store copies so a caller cannot mutate an already-stored transition.
        self.buffer.append(
            (
                state_array.copy(),
                int(action),
                float(reward),
                next_state_array.copy(),
                bool(terminated),
                bool(truncated),
            )
        )

    def sample(
        self,
        batch_size: int,
        *,
        include_truncation: bool = False,
    ) -> tuple[np.ndarray, ...]:
        """Sample a batch.

        By default, return the legacy five-array tuple, where the final array
        represents true termination only. New DQN code should set
        include_truncation=True to receive six arrays:
        states, actions, rewards, next_states, terminated, truncated.
        """
        size = int(batch_size)
        if size <= 0:
            raise ValueError("batch_size must be positive")
        if size > len(self.buffer):
            raise ValueError(f"Cannot sample {size} transitions from buffer of size {len(self.buffer)}")
        indices = self.rng.choice(len(self.buffer), size=size, replace=False)
        experiences = [self.buffer[int(index)] for index in indices]
        states, actions, rewards, next_states, terminated, truncated = zip(*experiences)
        result = (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(terminated, dtype=np.float32),
        )
        if include_truncation:
            return result + (np.asarray(truncated, dtype=np.float32),)
        return result

    def state_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot for later full-resume support."""
        transitions = [
            {
                "state": state.tolist(),
                "action": action,
                "reward": reward,
                "next_state": next_state.tolist(),
                "terminated": terminated,
                "truncated": truncated,
            }
            for state, action, reward, next_state, terminated, truncated in self.buffer
        ]
        return {
            "capacity": self.capacity,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "transitions": transitions,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a buffer snapshot, validating its dimensions and capacity."""
        expected = {
            "capacity": self.capacity,
            "state_size": self.state_size,
            "action_size": self.action_size,
        }
        actual = {key: state.get(key) for key in expected}
        if actual != expected:
            raise ValueError(f"Replay-buffer configuration mismatch: saved={actual}, current={expected}")
        transitions = state.get("transitions")
        if not isinstance(transitions, list) or len(transitions) > self.capacity:
            raise ValueError("Invalid transition list in replay-buffer state")
        self.buffer.clear()
        for item in transitions:
            self.push(
                item["state"],
                item["action"],
                item["reward"],
                item["next_state"],
                terminated=item["terminated"],
                truncated=item["truncated"],
            )
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


if __name__ == "__main__":
    buffer = ReplayBuffer(capacity=1000, state_size=18, action_size=6, seed=42)
    for i in range(100):
        state = np.zeros(18, dtype=np.float32)
        next_state = np.ones(18, dtype=np.float32)
        buffer.push(
            state,
            i % 6,
            1.0,
            next_state,
            terminated=(i == 99),
            truncated=(i == 98),
        )
    batch = buffer.sample(32, include_truncation=True)
    print("Buffer size:", len(buffer))
    print("Batch shapes:", [array.shape for array in batch])
    print("Replay buffer test passed.")
