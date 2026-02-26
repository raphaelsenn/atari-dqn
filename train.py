from argparse import Namespace, ArgumentParser

from dqn.dqn import AtariDQN
from dqn.env import make_atari_env


def parse_args() -> Namespace:
    # Default settings for SpaceInvaders
    parser = ArgumentParser()
    parser.add_argument("--env_id", type=str, default="ALE/SpaceInvaders-v5")
    parser.add_argument("--state_ch", type=int, default=4)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--buffer_capacity", type=int, default=1_000_000)
    parser.add_argument("--num_timesteps", type=int, default=50_000_000)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--exploration_initial_eps", type=float, default=1.0)
    parser.add_argument("--exploration_final_eps", type=float, default=0.1)
    parser.add_argument("--exploration_decay_steps", type=int, default=1_000_000)
    parser.add_argument("--update_network_every", type=int, default=4)
    parser.add_argument("--update_target_network_every", type=int, default=10_000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--save_every", type=int, default=1_000_000)
    parser.add_argument("--eval_every", type=int, default=50_000)
    return parser.parse_args() 


def main() -> None:
    args = parse_args()
    dqn = AtariDQN(
        state_ch=args.state_ch,
        action_dim=args.action_dim,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        learning_rate=args.learning_rate,
        exploration_inital_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        exploration_decay_steps=args.exploration_final_eps,
        update_network_every=args.update_network_every,
        update_target_network_every=args.update_target_network_every, 
        gamma=args.gamma,
        verbose=args.verbose,
        save_every=args.save_every,
        eval_every=args.eval_every,
        device=args.device
        ) 
    env = make_atari_env(args.env_id)
    dqn.learn(env, args.num_timesteps)


if __name__ == "__main__":
    main()