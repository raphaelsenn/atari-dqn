import sys
import os

import numpy as np
import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dqn.replay_buffer import ReplayBuffer


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def buffer(device):
    return ReplayBuffer(state_ch=4, buffer_capacity=8, device=device)


def make_transition(state_ch=4, val=0, done=False):
    s = np.full((state_ch, 84, 84), fill_value=val, dtype=np.uint8)
    s_nxt = np.full((state_ch, 84, 84), fill_value=(val + 1) % 256, dtype=np.uint8)
    a = int(val % 6)
    r = int([-1, 0, 1][val % 3]) 
    return s, a, r, s_nxt, done


def test_shapes_and_dtypes(buffer):
    assert buffer.states.shape == (buffer.buffer_capacity, buffer.state_ch, 84, 84)
    assert buffer.states_nxt.shape == (buffer.buffer_capacity, buffer.state_ch, 84, 84)
    assert buffer.actions.shape == (buffer.buffer_capacity,)
    assert buffer.rewards.shape == (buffer.buffer_capacity,)
    assert buffer.dones.shape == (buffer.buffer_capacity,)

    assert buffer.states.dtype == np.uint8
    assert buffer.states_nxt.dtype == np.uint8
    assert buffer.actions.dtype == np.uint8
    assert buffer.rewards.dtype == np.int8
    assert buffer.dones.dtype == np.uint8

    assert len(buffer) == 0
    assert buffer.size == 0
    assert buffer.position == 0


def test_push_1(buffer):
    s, a, r, s_nxt, done = make_transition(val=7, done=True)
    buffer.push(s, a, r, s_nxt, done)

    assert len(buffer) == 1
    assert buffer.size == 1
    assert buffer.position == 1  # started at 0

    # written at index 0
    assert np.array_equal(buffer.states[0], s)
    assert np.array_equal(buffer.states_nxt[0], s_nxt)
    assert int(buffer.actions[0]) == a
    assert int(buffer.rewards[0]) == r
    assert int(buffer.dones[0]) == 1


def test_push_2(buffer):
    cap = buffer.buffer_capacity

    # Fill the buffer
    for i in range(cap):
        s, a, r, s_nxt, done = make_transition(val=i, done=(i % 2 == 0))
        buffer.push(s, a, r, s_nxt, done)

    assert buffer.size == cap
    assert buffer.position == 0

    # Test if buffer overrides oldest 
    s_new, a_new, r_new, s_nxt_new, done_new = make_transition(val=123, done=True)
    buffer.push(s_new, a_new, r_new, s_nxt_new, done_new)

    assert buffer.size == cap
    assert buffer.position == 1
    assert np.array_equal(buffer.states[0], s_new)
    assert np.array_equal(buffer.states_nxt[0], s_nxt_new)
    assert int(buffer.actions[0]) == a_new
    assert int(buffer.rewards[0]) == r_new
    assert int(buffer.dones[0]) == 1


def test_sample_minibatch_1(buffer, device):
    # Add some transitions
    for i in range(5):
        s, a, r, s_nxt, done = make_transition(val=i * 10, done=(i % 2 == 1))
        buffer.push(s, a, r, s_nxt, done)

    batch_size = 4
    s, a, r, s_nxt, done = buffer.sample_minibatch(batch_size)

    # Samples should be torch.Tensor
    assert isinstance(s, torch.Tensor)
    assert isinstance(a, torch.Tensor)
    assert isinstance(r, torch.Tensor)
    assert isinstance(s_nxt, torch.Tensor)
    assert isinstance(done, torch.Tensor)

    # Check shapes
    assert s.shape == (batch_size, buffer.state_ch, 84, 84)
    assert s_nxt.shape == (batch_size, buffer.state_ch, 84, 84)
    assert a.shape == (batch_size,)
    assert r.shape == (batch_size,)
    assert done.shape == (batch_size,)

    # Check dtypes
    assert s.dtype == torch.float32
    assert s_nxt.dtype == torch.float32
    assert a.dtype == torch.int64
    assert r.dtype == torch.float32
    assert done.dtype == torch.uint8

    # Check device
    assert s.device == device
    assert s_nxt.device == device
    assert a.device == device
    assert r.device == device
    assert done.device == device

    # Check if pixels are normalizes
    assert float(s.min()) >= 0.0
    assert float(s.max()) <= 1.0
    assert float(s_nxt.min()) >= 0.0
    assert float(s_nxt.max()) <= 1.0

    # done must be 0/1
    assert torch.all((done == 0) | (done == 1))


def test_sample_minibatch_2(buffer):
    # size=2 is the minimum to sample
    for i in range(2):
        s, a, r, s_nxt, done = make_transition(val=i, done=False)
        buffer.push(s, a, r, s_nxt, done)

    s, a, r, s_nxt, done = buffer.sample_minibatch(batch_size=999)
    assert s.shape[0] == 2
    assert a.shape[0] == 2