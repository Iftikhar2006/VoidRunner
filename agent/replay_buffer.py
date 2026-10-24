import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(
            *experiences
        )

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    buffer = ReplayBuffer(capacity=1000)

    for i in range(100):
        state = np.zeros(8, dtype=np.float32)
        next_state = np.ones(8, dtype=np.float32)

        buffer.push(
            state=state,
            action=i % 6,
            reward=1.0,
            next_state=next_state,
            done=False,
        )

    batch = buffer.sample(32)

    print("Buffer size:", len(buffer))
    print("States shape:", batch[0].shape)
    print("Actions shape:", batch[1].shape)
    print("Rewards shape:", batch[2].shape)
    print("Next states shape:", batch[3].shape)
    print("Dones shape:", batch[4].shape)
