import random
from functools import partial
from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np
import torch
import wandb
from torch import nn as nn
from rich import print
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

from common import networks
from common.replay_buffer import UniformReplayBuffer, PrioritizedReplayBuffer
from common.utils import prep_observation_for_qnet
from common.pedqn import QuantileHuberLoss

class DLTV:
    buffer: Union[UniformReplayBuffer, PrioritizedReplayBuffer]

    def __init__(self, env, args: SimpleNamespace) -> None:
        self.env = env
        self.save_dir = args.save_dir
        self.use_amp = args.use_amp

        net = networks.get_model(args.network_arch, args.spectral_norm)
        linear_layer = nn.Linear
        depth = args.frame_stack*(1 if args.grayscale else 3)
        self.q_policy = net(depth, env.action_space.n, linear_layer, prior_beta=args.prior_beta, n_heads=args.n_heads).cuda()
        self.q_target = net(depth, env.action_space.n, linear_layer, prior_beta=args.prior_beta, n_heads=args.n_heads).cuda()
        self.q_target.load_state_dict(self.q_policy.state_dict())

        self.prior_beta = args.prior_beta
        self.beta = 0 #dummy
        self.n_heads = args.n_heads

        self.quantile_loss = QuantileHuberLoss(sum_over_quantiles=True, use_huber=True, order=False)
        self.p = torch.ones(size=(self.n_heads,)).cuda()/self.n_heads

        k = 0
        for parameter in self.q_policy.parameters():
           k += parameter.numel()
        print(f'Q-Network has {k} parameters.')

        self.double_dqn = args.double_dqn

        self.prioritized_er = args.prioritized_er
        if self.prioritized_er:
            self.buffer = PrioritizedReplayBuffer(args.burnin, args.buffer_size, args.gamma, args.n_step, args.parallel_envs, use_amp=self.use_amp)
        else:
            self.buffer = UniformReplayBuffer(args.burnin, args.buffer_size, args.gamma, args.n_step, args.parallel_envs, use_amp=self.use_amp)

        self.n_step_gamma = args.gamma ** args.n_step

        self.max_grad_norm = args.max_grad_norm
        self.opt = torch.optim.Adam(self.q_policy.parameters(), lr=args.lr, eps=args.adam_eps)
        self.scaler = GradScaler(enabled=self.use_amp)

        self.decay_lr = args.lr_decay_steps is not None
        if self.decay_lr: self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, (args.lr_decay_steps*args.train_count)//args.parallel_envs, gamma=args.lr_decay_factor)

    def sync_Q_target(self) -> None:
        self.q_target.load_state_dict(self.q_policy.state_dict())

    def set_beta (self, beta: float) -> None:
        self.beta = beta

    def act(self, states, eps: float):
        """ computes an epsilon-greedy step with respect to the current policy self.q_policy """
        with torch.no_grad():
            with autocast(enabled=self.use_amp):
                states = prep_observation_for_qnet(torch.from_numpy(np.stack(states)), self.use_amp)
                # (bs, n_heads, n_actions)
                z_dist = self.q_policy(states)
                
                q_mu = z_dist.mean(dim=1)

                ordered_atoms = torch.sort(z_dist, dim=1, descending=False)[0]
                
                # multiplication by 2 makes it easier to use same beta hp
                lt_var = 2 * torch.sum(1/self.n_heads * (ordered_atoms[:,self.n_heads//2:] - ordered_atoms[:,self.n_heads//2:self.n_heads//2+1])**2, dim=1)
                q_opt = q_mu + self.beta * lt_var
                actions = q_opt.argmax(dim=-1)

            if eps > 0:
                for i in range(actions.shape[0]):
                    if random.random() < eps:
                        actions[i] = self.env.action_space.sample()
            return actions.cpu()

    @torch.no_grad()
    def dtd_target(self, reward: float, next_state, done: bool):
        if self.double_dqn:
            z_dist = self.q_policy(next_state)
            q_mu = z_dist.mean(1)
            best_action = torch.argmax(q_mu, dim=1)

            next_Q = torch.gather(self.q_target(next_state), dim=2, index=best_action.view(-1,1,1).repeat(1,self.n_heads,1)).squeeze()
            return reward.unsqueeze(-1) + self.n_step_gamma * next_Q * (1 - done.unsqueeze(-1))
        else:
            raise NotImplementedError
            max_q = torch.max(self.q_target(next_state)[0], dim=2)[0]
            return reward.unsqueeze(-1) + self.n_step_gamma * max_q * (1 - done.unsqueeze(-1))

    def train(self, batch_size, beta=None) -> Tuple[float, float, float]:
        if self.prioritized_er:
            indices, weights, (state, next_state, action, reward, done) = self.buffer.sample(batch_size, beta)
            weights = torch.from_numpy(weights).cuda()
        else:
            state, next_state, action, reward, done = self.buffer.sample(batch_size)

        self.opt.zero_grad()
        with autocast(enabled=self.use_amp):
            z_dist_est = self.q_policy(state)
            dtd_est = torch.gather(z_dist_est, dim=2, index=action.view(-1,1,1).repeat(1,self.n_heads,1)).squeeze()
            dtd_tgt = self.dtd_target(reward, next_state, done)

            quant_loss = 0.1 * self.quantile_loss(dtd_est, self.p.view(1,-1).expand(dtd_tgt.shape), dtd_tgt).mean(-1)

            td_est = dtd_est.mean(1)
            td_tgt = dtd_tgt.mean(1)

            if self.prioritized_er:
                td_errors = (td_est-td_tgt)
                new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6  # 1e-6 is the epsilon in PER
                self.buffer.update_priorities(indices, new_priorities)

                loss = torch.mean(weights * quant_loss)
            else:
                loss = torch.mean(quant_loss)
                
        self.scaler.scale(loss).backward()

        self.scaler.unscale_(self.opt)
        grad_norm = nn.utils.clip_grad_norm_(list(self.q_policy.parameters()), self.max_grad_norm)
        self.scaler.step(self.opt)
        self.scaler.update()

        if self.decay_lr:
            self.scheduler.step()

        return td_est.mean().item(), loss.item(), grad_norm.item()

    def save(self, game_frame, **kwargs):
        save_path = (self.save_dir + f"/checkpoint_{game_frame}.pt")
        torch.save({**kwargs, 'state_dict': self.q_policy.state_dict(), 'game_frame': game_frame}, save_path)

        try:
            artifact = wandb.Artifact('saved_model', type='model')
            artifact.add_file(save_path)
            wandb.run.log_artifact(artifact)
            print(f'Saved model checkpoint at {game_frame} frames.')
        except Exception as e:
            print('[bold red] Error while saving artifacts to wandb:', e)
