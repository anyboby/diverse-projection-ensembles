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
from common.pedqn import CategoricalLoss

class IDS:
    buffer: Union[UniformReplayBuffer, PrioritizedReplayBuffer]

    def __init__(self, env, args: SimpleNamespace) -> None:
        self.env = env
        self.save_dir = args.save_dir
        self.use_amp = args.use_amp

        net = networks.get_model(args.network_arch, args.spectral_norm)
        linear_layer = nn.Linear
        depth = args.frame_stack*(1 if args.grayscale else 3)
        self.q_policy = net(depth, env.action_space.n, linear_layer, ensemble_size=args.ensemble_size, prior_beta=args.prior_beta, n_heads=args.n_heads).cuda()
        self.q_target = net(depth, env.action_space.n, linear_layer, ensemble_size=args.ensemble_size, prior_beta=args.prior_beta, n_heads=args.n_heads).cuda()
        self.q_target.load_state_dict(self.q_policy.state_dict())

        self.ensemble_size = args.ensemble_size
        self.prior_beta = args.prior_beta
        self.beta = 0 #dummy
        self.q_min, self.q_max = args.q_min, args.q_max
        self.n_heads = args.n_heads
        
        self.z = (self.q_min + (torch.arange(0.0, args.n_heads, dtype=torch.float16)) * (self.q_max - self.q_min) / float(args.n_heads - 1)).cuda()
        self.categorical_loss = CategoricalLoss()

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

        loss_fn_cls = nn.MSELoss if args.loss_fn == 'mse' else nn.SmoothL1Loss
        self.loss_fn = loss_fn_cls(reduction=('none' if self.prioritized_er else 'mean'))

    def sync_Q_target(self) -> None:
        self.q_target.load_state_dict(self.q_policy.state_dict())

    def set_beta (self, beta: float) -> None:
        self.beta = beta

    def act(self, states, eps: float):
        """ computes an epsilon-greedy step with respect to the current policy self.q_policy """
        with torch.no_grad():
            with autocast(enabled=self.use_amp):
                states = prep_observation_for_qnet(torch.from_numpy(np.stack(states)), self.use_amp)
                
                q_values, z_dist_r = self.q_policy(states)
                q_mu = q_values.mean(dim=1)
                q_std = q_values.std(dim=1)

                # expected regret
                # (bs, n_acts)
                delta_hat = ((q_mu + self.beta * q_std).max(dim=-1)[0]).unsqueeze(-1) - (q_mu - self.beta * q_std)

                # info_gain
                # (bs, n_heads, n_acts)
                z_dist_p = F.softmax(z_dist_r, dim=1)
                z = self.z.view(1, -1, 1).expand(z_dist_p.shape)
                z_mu = torch.sum(z * z_dist_p, dim=1, keepdim=True)
                z_var = torch.sum(z_dist_p * (z - z_mu)**2, dim=1)

                # for normalization
                z_var_partition = 1/self.env.action_space.n * torch.sum(z_var, dim=-1).unsqueeze(-1)
                z_var_norm = (z_var / (z_var_partition)).squeeze(1)

                # # clipping, done in original implementation
                z_var_norm = torch.clip(z_var_norm, min=0.25)

                # info gain
                i_gain = torch.log(1.0 + (q_std**2) / z_var_norm) + 1e-5

                ids_a = ((delta_hat**2) / i_gain)
                actions = ids_a.argmin(dim=-1)

            if eps > 0:
                for i in range(actions.shape[0]):
                    if random.random() < eps:
                        actions[i] = self.env.action_space.sample()
            return actions.cpu()

    @torch.no_grad()
    def td_target(self, reward: float, next_state, done: bool):
        if self.double_dqn:
            best_action = torch.argmax(self.q_policy(next_state)[0], dim=2)
            next_Q = torch.gather(self.q_target(next_state)[0], dim=2, index=best_action.unsqueeze(-1)).squeeze()
            return reward.unsqueeze(-1) + self.n_step_gamma * next_Q * (1 - done.unsqueeze(-1))
        else:
            max_q = torch.max(self.q_target(next_state)[0], dim=2)[0]
            return reward.unsqueeze(-1) + self.n_step_gamma * max_q * (1 - done.unsqueeze(-1))

    def dtd_target(self, reward: float, next_state, done: bool):
        if self.double_dqn:
            # (bs, n_heads, actions)
            raw = self.q_policy(next_state)[1]
            p = F.softmax(raw, dim=1)
            z = self.z.view(1, -1, 1).expand(p.shape)
            # (bs, actions)
            q_tp1_4a = torch.sum(p*z, dim=1)
            action_tp1 = torch.argmax(q_tp1_4a, dim=1)

            raw_target = self.q_target(next_state)[1]
            p_target = F.softmax(raw_target, dim=1)
            # (bs, n_heads)
            p_tp1_target = torch.gather(p_target, dim=2, index=action_tp1.view(-1,1,1).repeat(1,self.n_heads,1)).squeeze()
            z_tp1 = reward.view(-1,1) + self.n_step_gamma * (1-done.view(-1,1)) * z[...,0]

            return p_tp1_target, z_tp1
        else:
            raise NotImplementedError

    def train(self, batch_size, beta=None) -> Tuple[float, float, float]:
        if self.prioritized_er:
            indices, weights, (state, next_state, action, reward, done) = self.buffer.sample(batch_size, beta)
            weights = torch.from_numpy(weights).cuda()
        else:
            state, next_state, action, reward, done = self.buffer.sample(batch_size)

        self.opt.zero_grad()
        with autocast(enabled=self.use_amp):
            q_est, z_est = self.q_policy(state)
            td_est = torch.gather(q_est, dim=2, index=action.view(-1,1,1).repeat(1,self.ensemble_size,1)).squeeze()
            td_tgt = self.td_target(reward, next_state, done)

            r_dtd_est = torch.gather(z_est, dim=2, index=action.view(-1,1,1).repeat(1,self.n_heads,1)).squeeze()
            dtd_tgt_p, dtd_tgt_l = self.dtd_target(reward, next_state, done)
            
            # (bs,)
            cat_loss = self.categorical_loss(r_dtd_est, 
                                              dtd_tgt_p, 
                                              dtd_tgt_l, 
                                              self.q_min, # proj. lower
                                              self.q_max, # proj. upper
                                              self.n_heads,
                                              dist_dim=1)

            if self.prioritized_er:
                td_errors = (td_est-td_tgt).mean(-1)
                new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6  # 1e-6 is the epsilon in PER
                self.buffer.update_priorities(indices, new_priorities)

                q_losses = self.loss_fn(td_tgt, td_est).mean(-1)
                loss = torch.mean(weights * (q_losses + cat_loss))
            else:
                q_losses = self.loss_fn(td_tgt, td_est).mean(-1)
                loss = torch.mean(q_losses + cat_loss)
                
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
