from typing import Tuple

import torch
import numpy as np


class ReplayBuffer:
    """
    Replay buffer saves tuples (s_{t}, a_{t}, r_{t}, done_{t}) where:
    s_{t} : np.ndarray of shape [C, 84, 84] of type np.uint8
    a_{t} : uint8 (0, 1, ..., action_dim - 1)
    r_{t} : int8 (-1 or 0 or +1)
    done_{t} : uint8 (0 or 1)

    Required memory size:
    ---------------------
    Assume C = state_ch = 4:

    bytes(s_t) = 4 * 84 * 84 * 1 byte                    = 28_224 byte
    bytes(s_{t+1}) = 4 * 84 * 84 * 1 byte                = 28_224 byte
    bytes(r_{t})                                         = 1 byte
    bytes(a_{t})                                         = 1 byte
    bytes(done_{t})                                      = 1 byte
    ------------------------------------------------------------------
    (+)                                                  = 56_451 byte

    => Per transition: 56_451 byte
    If buffer_capacity = 1_000_000  => Buffer needs 1_000_000 * 56_451 byte = 56_451_000_000 byte  < 57 gb
    If buffer_capacity = 500_000    => Buffer needs 500_000 * 56_451 byte   = 28_225_500_000 byte  < 29 gb
    If buffer_capacity = 250_000    => Buffer needs 250_000 * 56_451 byte   = 14_112_650_000 byte  < 15 gb
    """
    def __init__(self, state_ch: int, buffer_capacity: int, device: torch.device) -> None:
        self.state_ch = state_ch 
        self.buffer_capacity = buffer_capacity
        self.size = 0 
        self.position = 0
        self.device = device
        
        self.states = np.empty(shape=(self.buffer_capacity, self.state_ch, 84, 84), dtype=np.uint8)
        self.states_nxt = np.empty(shape=(self.buffer_capacity, self.state_ch, 84, 84), dtype=np.uint8)
        self.actions = np.empty(shape=(self.buffer_capacity,), dtype=np.uint8)
        self.rewards = np.empty(shape=(self.buffer_capacity,), dtype=np.int8)
        self.dones = np.empty(shape=(self.buffer_capacity,), dtype=np.uint8)

    def __len__(self) -> int:
        return self.size
    
    def push(self, s: np.ndarray, a: int, r: int, s_nxt: np.ndarray, done: bool) -> None:
        """Pushes tuple (s_{t}, a_{t}, r_{t}, s_{t+1}, done_{t}) on the buffer."""
        j = self.position

        self.states[j] = s
        self.actions[j] = a
        self.rewards[j] = r
        self.states_nxt[j] = s_nxt
        self.dones[j] = int(done)

        self.size = min(self.buffer_capacity, self.size + 1)
        self.position = (self.position + 1) % self.buffer_capacity

    def sample_minibatch(self, batch_size: int=32) -> Tuple:
        """Samples mini-batch (s_{t}, a_{t}, r_{t}, s_{t+1}, done_{t}) from buffer."""
        if self.size < 2:
            raise ValueError("Need at least two transitions to sample.")

        batch_size = min(batch_size, len(self))
        j = np.random.randint(0, self.size, size=(batch_size,))

        s = torch.from_numpy(self.states[j]).to(self.device, torch.float32).div_(255.0)
        a = torch.from_numpy(self.actions[j]).to(self.device, torch.int64)
        r = torch.from_numpy(self.rewards[j]).to(self.device, torch.float32)
        s_nxt = torch.from_numpy(self.states_nxt[j]).to(self.device, torch.float32).div_(255.0)
        done = torch.from_numpy(self.dones[j]).to(self.device, torch.uint8)

        return (s, a, r, s_nxt, done)