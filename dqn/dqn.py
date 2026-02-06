import csv
from itertools import count

import numpy as np

import torch
import torch.nn as nn

import gymnasium as gym

from dqn.replay_buffer import ReplayBuffer
from env import make_atari_env


class DQN(nn.Module):
    """
    Deep Q-Network described in the deepmind paper: "Playing Atari with Deep Reinforcement Learning".

    References:
    ---------- 
    Human-level control through deep reinforcement learning; Mnih, et al. (2015).
    https://www.nature.com/articles/nature14236
    """ 
    def __init__(self, state_ch: int, action_dim: int):
        super().__init__()

        self.state_ch = state_ch
        self.action_dim = action_dim

        self.q = nn.Sequential(
            # [batch, state_ch, 84, 84] -> [batch, 32, 18, 18] 
            nn.Conv2d(in_channels=state_ch, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(True),
            
            # [batch, 32, 18, 18] -> [batch, 64, 9, 9]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(True),
            
            # [batch, 64, 9, 9] -> [batch, 64, 7, 7]
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(True),
            
            # [batch, 64, 7, 7] -> [batch, 64 * 7 * 7] 
            nn.Flatten(start_dim=1),
            
            # [batch, 64 * 7 * 7] -> [batch, 512] 
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(True),
            
            # [batch, 512] -> [batch, action_dim] 
            nn.Linear(512, action_dim)
        ) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:            # [C, H, W], else [batch, C, H, W]
            x = x.unsqueeze(0)      # [1, C, H, W]
        return self.q(x)

    def predict(self, x: torch.Tensor | np.ndarray) -> tuple[int, float]:
        if isinstance(x, np.ndarray): 
             x = torch.from_numpy(x).to(torch.float32).div_(255.0) 
        if x.dim() == 3:
            x = x.unsqueeze(0)
        q_values = self.q(x).squeeze(0)             # [action_dim,]
        q_max = q_values.max()                      # [1,]
        actions = torch.nonzero(q_values == q_max)  # [a_j_1, a_j_2, ...]
        idx = torch.randint(0,  len(actions), size=(1,)).item()
        action = int(actions[idx].item())
        return action , q_max 


class AtariDQN:
    """Deep Q-Network wrapper designed to train on Atari 2600 games.""" 
    def __init__(
            self, 
            state_ch: int,
            action_dim: int,
            batch_size: int,
            buffer_capacity: int,
            replay_start_size: int=50_000,
            learning_rate: float=2.5e-4,
            exploration_inital_eps: float=1.0,
            exploration_final_eps: float=0.1,
            exploration_decay_steps: int=1_000_000,
            update_network_every: int=4,
            update_target_network_every: int=10_000,
            gamma: float=0.99,
            verbose: bool=True,
            save_every: int=250_000,
            eval_every: int=250_000,
            device: str="cpu" 
    ) -> None:
        self.state_ch = state_ch
        self.action_dim = action_dim
        self.batch_size = batch_size

        self.learning_rate = learning_rate 
        self.epsilon = exploration_inital_eps
        self.epsilon_final = exploration_final_eps 
        self.epsilon_decay = (exploration_final_eps / exploration_inital_eps)**(1/exploration_decay_steps)
        self.exploration_decay_steps = exploration_decay_steps 
        self.update_target_network_every = update_target_network_every
        self.update_frequency = update_network_every
        self.gamma = gamma
        
        self.device = torch.device(device)

        self.Q = DQN(state_ch, action_dim).to(device)
        self.Q_target = DQN(state_ch, action_dim).to(device)
        self.Q_target.load_state_dict(self.Q.state_dict())

        self.criterion = nn.HuberLoss()
        self.optimizer = torch.optim.RMSprop(self.Q.parameters(), self.learning_rate, alpha=0.95, eps=0.01)
        self.buffer = ReplayBuffer(state_ch, buffer_capacity, self.device)
        self.buffer_capacity = buffer_capacity 
        self.replay_start_size = replay_start_size

        self.verbose = verbose
        self.save_every = save_every
        self.eval_every = eval_every

    @torch.no_grad()
    def get_action(self, s: torch.Tensor, epsilon: float) -> int:
        """
        Returns discrete action a in {0, 1, 2, ..., action_dim - 1}
            
        With probability epsilon an action is selected acording to:
            uniform(0, 1, ..., action_dim - 1)
        
        else, (with probability 1 - epsilon) an action is selected acording to:
            argmax_{a} Q(s, a) (ties broken)
        """  
        if np.random.random() < epsilon:
            return int(np.random.randint(self.action_dim))

        if s.dim() == 3:                            # [C, W, H]
            s = s.unsqueeze(0)                      # [1, C, W, H]

        s = s.to(self.device)

        q_values = self.Q(s).squeeze(0)             # [action_dim,]
        q_max = q_values.max()                      # [1,]
        actions = torch.nonzero(q_values == q_max)  # [a_j_1, a_j_2, ...]
        idx = torch.randint(0,  len(actions), size=(1,), device=self.device).item()
        action = actions[idx]
        return action

    @torch.no_grad()
    def get_greedy_action(self, s: torch.Tensor) -> int:
        """
        Returns an discrete greedy action:
        argmax_{a} Q(s, a) (ties broken)
        """ 
        if s.dim() == 3:                            # [C, W, H]
            s = s.unsqueeze(0)                      # [1, C, W, H]
        
        s = s.to(self.device)

        q_values = self.Q(s).squeeze(0)             # [action_dim,]
        action = int(torch.argmax(q_values).item())
        return action
    
    def update_network(self) -> None:
        """
        Updates the deep Q-network parameters.

        Objective:
        ---------
        L_{i} = E[(r + gamma * max_{a'} Q(s', a', theta_{i - 1}) - Q(s, a, theta_{i}))]
        """ 
        # Sample minibatch
        phi, a, r, phi_nxt, done = self.buffer.sample_minibatch(self.batch_size)
        
        # Get predictions of Q-network
        q_pred = self.Q(phi).gather(1, a.unsqueeze(1)).squeeze(1)

        # Compute TD target using Q-target-network
        with torch.no_grad(): 
            q_max_nxt = self.Q_target(phi_nxt).max(dim=-1).values
            not_done = 1.0 - done.float() 
            y = r + not_done * self.gamma * q_max_nxt 

        # Compute mse loss
        loss = self.criterion(y, q_pred)

        # Update Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network (hard update)
        if self.t % self.update_target_network_every == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def learn(self, env: gym.Env, num_timesteps: int=10_000_000) -> None:
        """Training algorithm""" 
        self.env_id = env.spec.id 
        self.num_timesteps = num_timesteps

        self.t = 0                      # global steps
        self.report: list[dict] = []    # global report

        self.explore_env(self.replay_start_size)
        for e in count():
            s, info = env.reset()
            phi = torch.from_numpy(s).to(torch.float32).to(self.device).div_(255.0)
            done = False 

            while not done:
                current_life = info["lives"]

                a = self.get_action(phi, self.epsilon)
                s_nxt, r, terminated, truncated, info = env.step(a)
                r = np.sign(r)
                done = terminated or truncated
                done_td = True if ((info["lives"] < current_life) or done) else False

                # Increment step counter 
                self.t += 1

                # Update buffer
                self.buffer.push(s, a, r, s_nxt, done_td)

                # Update network and epsilon 
                if self.t % self.update_frequency == 0: 
                    self.update_network()
                self._decay_epsilon()
                
                s = s_nxt
                phi = torch.from_numpy(s).to(torch.float32).to(self.device).div_(255.0)

                if self.t % self.eval_every == 0:
                    episode_scores = self.eval_env(5)
                    mean_episode_score = np.mean(episode_scores)
                    self.report.append(
                        {"Episode": e, "Timestep": self.t, "Mean-Episode-Score": mean_episode_score, **{f"score_{i}" : episode_scores[i] for i in range(5)}})

                    if self.verbose:
                        print(f"Episode: {e}\tTimestep: {self.t}\tMean-Episode-Score: {mean_episode_score:.4f}")
            
                if self.t % self.save_every == 0:
                    self._checkpoint()
                
            if self.t >= num_timesteps:
                break

        self._checkpoint() 
        env.close()

    @torch.no_grad()
    def eval_env(self, eval_runs: int=5) -> list[float]:
        """Evaluation in online environment""" 
        env = make_atari_env(self.env_id, None, self.state_ch)
        episode_scores = []
        for _ in range(eval_runs):
            episode_score = 0.0 
            s, _ = env.reset()
            phi = torch.from_numpy(s).to(torch.float32).to(self.device).div_(255.0)

            done = False
            while not done:
                a = self.get_greedy_action(phi)
                s_nxt, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated 

                s = s_nxt
                phi = torch.from_numpy(s).to(torch.float32).to(self.device).div_(255.0)
                episode_score += r
            episode_scores.append(episode_score)
        env.close()
        return episode_scores

    def explore_env(self, num_steps: int) -> None:
        env = make_atari_env(self.env_id, None, self.state_ch)
        t = 0
        while t < num_steps: 
            s, info = env.reset()
            done = False

            while not done and t < num_steps:
                current_live = info["lives"] 
                
                a = int(np.random.randint(self.action_dim))
                s_nxt, r, terminated, truncated, info = env.step(a)
                r = np.sign(r) 
                done = terminated or truncated 
                done_td = True if ((info["lives"] < current_live) or done) else False

                # Update buffer
                self.buffer.push(s, a, r, s_nxt, done_td)
                s = s_nxt
                t += 1
        env.close()

    def save(self, file_name: str|None=None) -> None:
        save_name = self._default_ckpt_name() if file_name is None else file_name
        torch.save(self.Q.state_dict(), save_name)

    def _default_ckpt_name(self) -> str:
        env = getattr(self, "env_id", "unkown_env").replace("/", "-")
        lr = getattr(self, "learning_rate")
        t = getattr(self, "t")
        return f"DQN-{env}-lr{lr}-t{t}.pt"

    def _save_report_as_csv(self) -> None:
        env = getattr(self, "env_id", "unkown_env").replace("/", "-")
        lr = getattr(self, "learning_rate")
        save_name = f"DQN-{env}-lr{lr}-report.csv"
        with open(save_name, "w", newline="") as csvfile:
            fieldnames = ["Episode", "Timestep", "Mean-Episode-Score"] + [f"score_{i}" for i in range(5)]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.report)

    def _checkpoint(self) -> None:
        self._save_report_as_csv()
        self.save()

    def _decay_epsilon(self) -> None:
        if self.t < self.exploration_decay_steps:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_final