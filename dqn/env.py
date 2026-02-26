import numpy as np
import gymnasium as gym
import ale_py


class FireResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """Take action on reset for environments that are fixed until firing."""
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, {}


class AutoFireOnLifeLoss(gym.Wrapper):
    """Take action after live loss for environments that are fixed until firing (for Breakout).""" 
    def __init__(self, env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        self.fire_action = meanings.index("FIRE")
        self.lives = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.lives = info.get("lives", self.lives)
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        lives = info.get("lives", self.lives)
        if lives < self.lives and not (term or trunc):
            obs, r2, term, trunc, info2 = self.env.step(self.fire_action)
            r += r2
            info.update(info2)
        self.lives = lives
        return obs, r, term, trunc, info


def make_atari_env(env_id: str, render_mode: str|None=None, stack_frames: int=4) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode, frameskip=1)
    if "Breakout" in env_id: 
        env = AutoFireOnLifeLoss(env)
        env = FireResetEnv(env)
    env = gym.wrappers.AtariPreprocessing(env)
    env = gym.wrappers.FrameStackObservation(env, stack_frames)
    return env  