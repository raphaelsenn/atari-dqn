from argparse import Namespace, ArgumentParser

import gymnasium as gym

import numpy as np

import torch

from dqn.dqn import DQN
from env import make_atari_env
from visualize import Visualize


def play(
        env: gym.Env, 
        dqn: DQN, 
        n_episodes: int=10, 
        verbose: bool=True, 
        plotting: bool=False
) -> None:
    if plotting: 
        vis = Visualize(dqn.action_dim) 
    
    for episode in range(n_episodes): 
        timestep = 0
        s, _ = env.reset()

        done = False 
        total_reward = 0.0
        timestep = 0
        if plotting: vis.reset()

        while not done:
            a, q_s_a, q_values = dqn.predict(s)                 # types: int, float, np.ndarray

            s_nxt, r, terminated, truncated, _ = env.step(a)
            total_reward += np.sign(r)
            s = s_nxt
            done = terminated or truncated

            if plotting and timestep % 4 == 0:
                vis.update(q_values, q_s_a) 
            if verbose: 
                print(
                    f"------------------------------------------------\n"
                    f"Episode: {episode}  \tTimestep: {timestep}      \n"
                    f"Action: {a}         \tSource: NEURAL NET (DQN)  \n"
                    f"Q(s, a): {q_s_a:.2f}                            \n"
                    f"Reward: {r}         \tDone: {done}              \n"
                    f"Total Reward: {total_reward}                    \n"
                    f"------------------------------------------------\n"
                )
            timestep += 1


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--env_id", type=str, default="ALE/SpaceInvaders-v5")
    parser.add_argument("--state_ch", type=int, default=4)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--weights", type=str, default="DQN-ALE-SpaceInvaders-v5-lr0.00025-50Mio.pt")
    parser.add_argument("--plotting", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=True)
    return parser.parse_args() 


def main() -> None:
    args = parse_args()
    dqn = DQN(args.state_ch, args.action_dim)
    dqn.load_state_dict(torch.load(args.weights, map_location="cpu"))
    env = make_atari_env(args.env_id, "human", args.state_ch)
    play(env, dqn, verbose=args.verbose, plotting=args.plotting)


if __name__ == "__main__":
    main()