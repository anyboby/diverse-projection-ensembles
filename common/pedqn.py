import random
from functools import partial
from types import SimpleNamespace
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from rich import print
from torch import nn as nn
from torch.cuda.amp import GradScaler, autocast, custom_fwd

import wandb
from common import networks
from common.replay_buffer import PrioritizedReplayBuffer, UniformReplayBuffer
from common.utils import prep_observation_for_qnet, RunningMeanStdTh

class PEDQN:
    buffer: Union[UniformReplayBuffer, PrioritizedReplayBuffer]

    def __init__(self, env, args: SimpleNamespace) -> None:
        self.env = env
        self.save_dir = args.save_dir
        self.use_amp = args.use_amp
        self.verbosity = args.verbosity
    
        net = networks.get_model(args.network_arch, args.spectral_norm)
        linear_layer = nn.Linear
        depth = args.frame_stack*(1 if args.grayscale else 3)
        self.q_policy = net(depth, env.action_space.n, linear_layer, args.n_heads, prior_beta=args.prior_beta).cuda()
        self.q_target = net(depth, env.action_space.n, linear_layer, args.n_heads, prior_beta=args.prior_beta).cuda()
        self.q_target.load_state_dict(self.q_policy.state_dict())

        self.double_dqn = args.double_dqn

        self.norm_ri = args.normalize_ri
        if self.norm_ri:
            self.intr_rms = RunningMeanStdTh(shape=(1,)).cuda()
            self.mean_adjustment = args.mean_adjustment_ri
        self.ri_norm = args.ri_norm
        self.episodic_ri = args.episodic_ri

        self.q_min, self.u_min = args.q_min, -0.1/(1-args.gamma_i ** args.n_step) if self.norm_ri else 0   # heuristic
        self.q_max, self.u_max = args.q_max, 0.1/(1-args.gamma_i ** args.n_step) if self.norm_ri else (args.q_max-args.q_min)*2   # heuristic
        self.n_atoms = args.n_heads
        self.z = (self.q_min + (torch.arange(0.0, args.n_heads, dtype=torch.float16)) * (self.q_max - self.q_min) / float(args.n_heads - 1)).cuda()
        self.z_u = (self.u_min + (torch.arange(0.0, args.n_heads, dtype=torch.float16)) * (self.u_max - self.u_min) / float(args.n_heads - 1)).cuda()
        self.p = (torch.ones(size=(args.n_heads,), dtype=torch.float16 if self.use_amp else torch.float32)/args.n_heads).cuda()
        self.beta = 0 # only a dummy init value
        
        k = 0
        for parameter in self.q_policy.parameters():
           k += parameter.numel()
        print(f'Q-Network has {k} parameters.')

        self.prioritized_er = args.prioritized_er
        if self.prioritized_er:
            self.buffer = PrioritizedReplayBuffer(args.burnin, args.buffer_size, args.gamma, args.n_step, args.parallel_envs, use_amp=self.use_amp)
        else:
            self.buffer = UniformReplayBuffer(args.burnin, args.buffer_size, args.gamma, args.n_step, args.parallel_envs, use_amp=self.use_amp)

        self.n_step_gamma = args.gamma ** args.n_step
        self.n_step_gamma_i = args.gamma_i ** args.n_step
        self.burnin_u = args.burnin_uncertainty

        self.max_grad_norm = args.max_grad_norm
        self.opt = torch.optim.Adam(self.q_policy.parameters(), lr=args.lr, eps=args.adam_eps)
        self.scaler = GradScaler(enabled=self.use_amp)

        self.decay_lr = args.lr_decay_steps is not None
        if self.decay_lr: self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, (args.lr_decay_steps*args.train_count)//args.parallel_envs, gamma=args.lr_decay_factor)
        
        self.categorical_loss = CategoricalLoss()
        self.quantile_loss = QuantileHuberLoss(sum_over_quantiles=True, use_huber=True, order=False)

    def sync_Q_target(self) -> None:
        self.q_target.load_state_dict(self.q_policy.state_dict())

    def set_beta (self, beta: float) -> None:
        self.beta = beta

    @custom_fwd(cast_inputs=torch.float32)
    def plr(self, tensor: torch.Tensor):
        p = self.p.view(1, -1, 1).expand(tensor[:,0].shape)
        z = self.z.view(1, -1, 1).expand(tensor[:,0].shape)
        z_u = self.z_u.view(1, -1, 1).expand(tensor[:,0].shape)

        probs = torch.stack([F.softmax(tensor[:,0], dim=1), p, F.softmax(tensor[:,2], dim=1), p], dim=1)
        locs = torch.stack([z, tensor[:,1], z_u, tensor[:,3]], dim=1)
        raw = torch.stack([tensor[:,0], p, tensor[:,2], p], dim=1)

        return probs, locs, raw

    def act(self, states, eps: float):
        """ computes an epsilon-greedy step with respect to the current policy self.q_policy """
        with torch.no_grad():
            with autocast(enabled=self.use_amp):
                states = prep_observation_for_qnet(torch.from_numpy(np.stack(states)), self.use_amp)
                # (bs, n_models, n_heads, n_acts)
                pal = self.plr(self.q_policy(states))
                # (bs, n_models, n_heads, n_acts)
                p_q, l_q, _ = pal[0][:,0:2], pal[1][:,0:2], pal[2][:,0:2]
                p_u, l_u, _ = pal[0][:,2:4], pal[1][:,2:4], pal[2][:,2:4]
                # (bs, n_acts)
                action_values = torch.sum(p_q*l_q,dim=2).mean(dim=1)
                # bonus (bs, n_heads, n_acts), project quantiles first
                p_proj = project_cat(l_q[:,1], p_q[:,1], self.q_min, self.q_max, self.n_atoms, dist_dim=1)
                # (bs, n_heads, n_acts)
                cdf1, cdf2 = torch.cumsum(p_proj, dim=1), torch.cumsum(p_q[:,0], dim=1)
                # (bs, n_acts)
                bonus_ = torch.sum(torch.pow(torch.abs(cdf1-cdf2) * (((self.q_max-self.q_min)/(self.n_atoms-1))**self.ri_norm), 1/self.ri_norm), dim=1)
                if self.norm_ri:
                    bonus = (1-self.n_step_gamma_i) * self.intr_rms(bonus_, adjustmean=True, fractionalmean=self.mean_adjustment)
                else:
                    bonus=bonus_
                # (bs, n_acts)
                action_u_values = torch.sum(p_u*l_u,dim=2).mean(dim=1)
                # (bs)
                actions = torch.argmax(action_values + self.beta * (action_u_values + bonus), dim=1)
                if self.norm_ri:
                    self.intr_rms.update(torch.gather(bonus_, dim=1, index=actions.unsqueeze(-1)))
                
            if eps > 0:
                for i in range(actions.shape[0]):
                    if random.random() < eps:
                        actions[i] = self.env.action_space.sample()
            
            return actions.cpu()

    @torch.no_grad()
    def dtd_target(self, reward: float, next_state, done: bool):
        if self.double_dqn:
            p_tp1, l_tp1, r_tp1 = self.plr(self.q_policy(next_state))
            q_tp1 = torch.sum(p_tp1[:,0:2]*l_tp1[:,0:2], dim=2)
            action_tp1_ = torch.argmax(q_tp1.mean(dim=1), dim=1)
            action_tp1 = action_tp1_.view(-1,1,1,1).expand(-1, 2, self.n_atoms,1)
            
            pal_tp1 = self.plr(self.q_target(next_state))
            p_t_tp1_q, l_t_tp1_q, r_t_tp1_q = (torch.gather(t[:,0:2], dim=3, index=action_tp1).squeeze() for t in (pal_tp1))

            # mixture distribution
            p_t_tp1_qm = torch.flatten(p_t_tp1_q, start_dim=1, end_dim=2)/2
            l_t_tp1_qm = torch.flatten(l_t_tp1_q, start_dim=1, end_dim=2)
            
            l_t_tp1_qm = reward.view(-1,1) + self.n_step_gamma * (1-done.view(-1,1)) * l_t_tp1_qm
            
            # uncertainty estimates
            # project quantiles to categorical support
            p_proj_tp1 = project_cat(l_tp1[:,1], p_tp1[:,1], self.q_min, self.q_max, self.n_atoms, dist_dim=1)
            cdf1, cdf2 = torch.cumsum(p_proj_tp1, dim=1), torch.cumsum(p_tp1[:,0], dim=1)
            reward_i_tp1_ = torch.sum(torch.pow(torch.abs(cdf1-cdf2) * (((self.q_max-self.q_min)/(self.n_atoms-1))**self.ri_norm), 1/self.ri_norm), dim=1)
            if self.norm_ri:
                reward_i_tp1 = (1-self.n_step_gamma_i) * self.intr_rms(reward_i_tp1_, adjustmean=True, fractionalmean=self.mean_adjustment)

            u_tp1 = torch.sum(p_tp1[:,2:4]*l_tp1[:,2:4], dim=2).mean(1)
            action_tp1_u_ = torch.argmax(q_tp1.mean(dim=1) + self.beta * (u_tp1 + reward_i_tp1), dim=1)
            reward_i_tp1_sel = torch.gather(reward_i_tp1, dim=1, index=action_tp1_u_.view(-1,1))
            
            action_tp1_u = action_tp1_u_.view(-1,1,1,1).expand(-1, 2, self.n_atoms,1)
            p_t_tp1_u, l_t_tp1_u, r_t_tp1_u = (torch.gather(t[:,2:4], dim=3, index=action_tp1_u).squeeze() for t in (pal_tp1))

            # mixture distribution
            p_t_tp1_um = torch.flatten(p_t_tp1_u, start_dim=1, end_dim=2)/2
            l_t_tp1_um = torch.flatten(l_t_tp1_u, start_dim=1, end_dim=2)
            
            if self.episodic_ri:
                l_t_tp1_um = self.n_step_gamma_i * (1-done.view(-1,1)) * (l_t_tp1_um + reward_i_tp1_sel)
            else:
                l_t_tp1_um = self.n_step_gamma_i * (l_t_tp1_um + reward_i_tp1_sel)

            diag = dict(reward_i = reward_i_tp1_sel.squeeze(), action_tp1=action_tp1_, opt_action_tp1=action_tp1_u_)
            return torch.stack((p_t_tp1_qm, p_t_tp1_um), dim=1), torch.stack((l_t_tp1_qm, l_t_tp1_um), dim=1), diag
        else:
            raise NotImplementedError
            max_q = torch.max(self.q_target(next_state), dim=1)[0]
            return reward + self.n_step_gamma * max_q * (1 - done)

    def train(self, batch_size, beta=None) -> Tuple[float, float, float]:
        if self.prioritized_er:
            indices, weights, (state, next_state, action, reward, done) = self.buffer.sample(batch_size, beta)
            weights = torch.from_numpy(weights).cuda()
        else:
            state, next_state, action, reward, done = self.buffer.sample(batch_size)

        self.opt.zero_grad()
        with autocast(enabled=self.use_amp):
            # (bs) -> (bs, n_models, n_heads, n_acts)
            action = action.view(-1,1,1,1).expand(-1, 4, self.q_policy.n_heads,1)
            p, l, r = self.plr(self.q_policy(state))
            # (bs, n_models, n_heads, n_acts) -> (bs, n_models, n_heads)
            p_td_est = torch.gather(p, dim=3, index=action).squeeze()
            l_td_est = torch.gather(l, dim=3, index=action).squeeze()

            # (bs, n_models, n_heads)
            p_dtd_tgt, l_dtd_tgt, tgt_diag = self.dtd_target(reward, next_state, done)
            # dtd_tgt contains tuple with shapes:
            #   ((bs, 2, n_heads), (bs, 2, n_heads)) 
            #   where the first item is probabilities of (Q-mixture, U-mixture), second item is locations of (Q-mixture, U-mixture)
            
            # -> (bs)
            td_tgt = torch.sum(p_dtd_tgt[:,0] * l_dtd_tgt[:,0], dim=1, keepdim=True)
            # -> (bs)
            td_est = torch.sum(p_td_est[:,0:2] * l_td_est[:,0:2], dim=2)

            # pick only categorical models for raw logits
            r_dtd_est = torch.gather(r[:,::2], dim=3, index=action[:,::2]).squeeze()
            
            with torch.no_grad():
                lower = l_td_est[:,0::2,0:1].expand(l_dtd_tgt.shape)
                upper = l_td_est[:,0::2,-1:].expand(l_dtd_tgt.shape)

            # get cross entropy loss for categorical models
            cat_loss_ = self.categorical_loss(r_dtd_est,
                                              p_dtd_tgt,
                                              l_dtd_tgt,
                                              lower, # proj. lower
                                              upper, # proj. upper
                                              self.n_atoms)
            cat_loss_[:,0] = cat_loss_[:,0]
            cat_loss_[:,1] = cat_loss_[:,1]

            if self.buffer.size>self.burnin_u:
                cat_loss = cat_loss_.mean(1)
            else:
                cat_loss = cat_loss_[:,0]

            # get quantile regression loss for quantile models
            quant_loss_ = self.quantile_loss(l_td_est[:,1::2], p_dtd_tgt, l_dtd_tgt).mean(-1)
            quant_loss_[:,0] = quant_loss_[:,0]
            quant_loss_[:,1] = quant_loss_[:,1]

            if self.buffer.size>self.burnin_u:
                quant_loss = quant_loss_.mean(1)
            else:
                quant_loss = quant_loss_[:,0]

            losses = cat_loss + quant_loss
        
            if self.prioritized_er:
                td_errors = td_est-td_tgt
                new_priorities = np.abs(td_errors.mean(1).detach().cpu().numpy()) + 1e-6  # 1e-6 is the epsilon in PER
                self.buffer.update_priorities(indices, new_priorities)
                loss = torch.mean(weights * losses)
            else:
                loss = torch.mean(losses)

        self.scaler.scale(loss).backward()

        self.scaler.unscale_(self.opt)
        grad_norm = nn.utils.clip_grad_norm_(list(self.q_policy.parameters()), self.max_grad_norm)
        self.scaler.step(self.opt)
        self.scaler.update()

        if self.decay_lr:
            self.scheduler.step()

        # collect diagnostics
        diagnostics = {}
        if self.verbosity>1:
            with autocast(enabled=self.use_amp):
                with torch.no_grad():
                    # compute mean and var
                    z_mean = torch.sum(p_td_est * l_td_est, dim=2)
                    z_var = torch.sum(p_td_est * (l_td_est - z_mean.unsqueeze(-1))**2, dim=2)
                    # compute churn
                    action_tp1 = tgt_diag['action_tp1']
                    opt_action_tp1 = tgt_diag['opt_action_tp1']
                    p_tp1_n, l_tp1_n, r_tp1_n = self.plr(self.q_policy(next_state))
                    q_tp1_n = torch.sum(p_tp1_n[:,0:2]*l_tp1_n[:,0:2], dim=2)
                    action_tp1_n = torch.argmax(q_tp1_n.mean(dim=1), dim=1)
                    p_proj_tp1_n = project_cat(l_tp1_n[:,1], p_tp1_n[:,1], self.q_min, self.q_max, self.n_atoms, dist_dim=1)
                    cdf1_n, cdf2_n = torch.cumsum(p_proj_tp1_n, dim=1), torch.cumsum(p_tp1_n[:,0], dim=1)
                    reward_i_tp1_n = torch.sum(torch.abs(cdf1_n-cdf2_n)*((self.q_max-self.q_min)/(self.n_atoms-1)), dim=1)
                    u_tp1_n = torch.sum(p_tp1_n[:,2:4]*l_tp1_n[:,2:4], dim=2).mean(1)
                    opt_action_tp1_n = torch.argmax(q_tp1_n.mean(dim=1) + (u_tp1_n + reward_i_tp1_n), dim=1)
                    
                    #log 
                    diagnostics = dict(
                        q_values_c = z_mean[:,0].mean().item(),
                        q_values_q = z_mean[:,1].mean().item(),
                        q_max_c = z_mean[:,0].max().item(),
                        q_max_q = z_mean[:,1].max().item(),
                        z_var_c = z_var[:,0].mean().item(),
                        z_var_q = z_var[:,1].mean().item(),
                        q_loss_c = cat_loss_[:,0].mean().item(),
                        q_loss_q = quant_loss_[:,0].mean().item(),
                        u_values_c = z_mean[:,2].mean().item(),
                        u_values_q = z_mean[:,3].mean().item(),
                        u_max_c = z_mean[:,2].max().item(),
                        u_max_q = z_mean[:,3].max().item(),
                        u_min_c = z_mean[:,2].min().item(),
                        u_min_q = z_mean[:,3].min().item(),
                        zu_var_c = z_var[:,2].mean().item(),
                        zu_var_q = z_var[:,3].mean().item(),
                        u_loss_c = cat_loss_[:,1].mean().item(),
                        u_loss_q = quant_loss_[:,1].mean().item(),
                        reward_i = tgt_diag['reward_i'].mean().item(),
                        reward_i_max = tgt_diag['reward_i'].max().item(),
                        reward_i_min = tgt_diag['reward_i'].min().item(),
                        optimistic_overturn = (torch.sum(opt_action_tp1 != action_tp1)/len(action_tp1)).item(),
                        optimistic_churn = (torch.sum(opt_action_tp1 != opt_action_tp1_n)/len(opt_action_tp1)).item(),
                        online_churn = (torch.sum(action_tp1 != action_tp1_n)/len(action_tp1)).item(),
                    )

        return td_est.mean().item(), loss.item(), grad_norm.item(), diagnostics

    def save(self, game_frame, **kwargs):
        save_path = (self.save_dir + f"/checkpoint_{game_frame}.pt")
        torch.save({**kwargs, 'state_dict': self.q_policy.state_dict(), 'game_frame': game_frame, 'intr_rms_state_dict': self.intr_rms.state_dict()}, save_path)

        try:
            artifact = wandb.Artifact('saved_model', type='model')
            artifact.add_file(save_path)
            wandb.run.log_artifact(artifact)
            print(f'Saved model checkpoint at {game_frame} frames.')
        except Exception as e:
            print('[bold red] Error while saving artifacts to wandb:', e)

class CategoricalLoss(nn.Module):
    """
    KL divergence loss between projected categorical distributions.
    """

    def __init__(self,) -> None:
        super(CategoricalLoss, self).__init__()

    def forward(self, 
                dtd_est_rawp: torch.Tensor, 
                p_dtd_tgt: torch.Tensor, 
                l_dtd_tgt: torch.Tensor, 
                lower: torch.Tensor, 
                upper: torch.Tensor, 
                n_atoms : int,
                dist_dim: int=2) -> torch.Tensor:
        """
        args:
            :param dtd_tgt:
                Expects distributional targets to have tuples with shape:
                Probabilitities: (bs, n_models, n_atoms), Locations: (bs, n_models, n_atoms)
            :param dtd_est_raw: 
                Pre softmax logits of the distributional estimates with shape
                Logits: (bs, n_models, n_atoms)
        """        
        # project: -> (bs, n_models, n_atoms)
        proj_tgt_p = project_cat(
            probs = p_dtd_tgt,
            locs = l_dtd_tgt,
            lower = lower,
            upper = upper,
            n_atoms = n_atoms,
            dist_dim = dist_dim,
            )
        # compute cross entropy loss from raw logits (numerically more stable and faster than log(softmax))
        # -> (bs, n_models)
        celoss = torch.sum(-proj_tgt_p.detach() * F.log_softmax(dtd_est_rawp, -1), -1)
        return celoss

def project_cat(locs, probs, lower, upper, n_atoms, dist_dim) -> torch.Tensor:
    """
    Projects sampled locations onto an interval of fixed locations between min and max. 
    Shapes should be:
        :param target_locs: (batch_size, nensemble, K, nactions)
        :param target_probs: (batch_size, nensemble, K, nactions)
        :param min/max: (,)
        :param dist_dim: 2
    where dist_dim is 2 because the number of bins are contained at dimension w. index 2
    """
    # project quantiles onto cat dist.
    # clamp between qmin and qmax
    # -> (bs, n_ens, n_heads_tar,?)
    q_tau = torch.clamp(locs, lower, upper)

    b = (q_tau - lower) / (   (upper - lower) / float(n_atoms - 1)   )

    ## for numerical stability, ceil can behave odd on floating point integers"
    b = torch.clamp(b, 0, n_atoms-1)

    lb = torch.floor(b)
    ub = torch.ceil(b)

    # when b happens to be an integer, lb == ub, so pr_j(s', a*) will
    # be discarded because (ub-b) == (b-lb) == 0.
    # (bs, n_ens, n_atoms_tar, ?)
    floor_equal_ceil = ((ub - lb) < 0.5).float()

    # (bs, n_ens, n_heads_tar, ?, n_heads)
    l_project_b = F.one_hot(lb.long(), n_atoms)

    # (bs, n_ens, n_heads_tar, ?, n_heads)
    u_project_b = F.one_hot(ub.long(), n_atoms)

    # adjust shape of one-hot encoding
    # (bs, n_ens, n_heads_tar, n_heads, ?)
    if dist_dim<len(probs.shape)-1:
        l_project_b = l_project_b.swapaxes(-1, -2)
        u_project_b = u_project_b.swapaxes(-1, -2)

    # -> (bs, n_ens, n_heads)
    ml_delta = probs * (ub - b + floor_equal_ceil)
    mu_delta = probs * (b - lb)

    # -> (bs, n_ens, n_heads, n_heads, ?) * (bs, n_ens, n_heads, 1, ?)
    #   -> (bs, n_ens, n_heads, n_heads, ?) -> (bs, n_ens, n_heads, ?)
    ml_delta = torch.sum(l_project_b * torch.unsqueeze(ml_delta, dist_dim+1), dim=dist_dim)
    mu_delta = torch.sum(u_project_b * torch.unsqueeze(mu_delta, dist_dim+1), dim=dist_dim)

    m = ml_delta + mu_delta

    return m

class QuantileHuberLoss(nn.Module):
    def __init__(self, sum_over_quantiles: bool = True, use_huber: bool = True, order: bool = False) -> None:
        super(QuantileHuberLoss, self).__init__()
        self.use_huber = use_huber
        self.sum_over_quantiles = sum_over_quantiles
        self.order = order

    def forward(self, 
        dtd_est_locs: torch.Tensor,
        p_dtd_tgt: torch.Tensor,
        l_dtd_tgt: torch.Tensor,
        cum_prob: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        The quantile-regression loss, as described in the QR-DQN and TQC papers.
        Partially taken from https://github.com/bayesgroup/tqc_pytorch.

        :param current_quantiles: current estimate of quantiles, must be either
            (batch_size, n_quantiles) or (batch_size, n_critics, n_quantiles)
        :param target_quantiles: target of quantiles, must be either (batch_size, n_target_quantiles),
            (batch_size, 1, n_target_quantiles), or (batch_size, n_critics, n_target_quantiles)
        :param cum_prob: cumulative probabilities to calculate quantiles (also called midpoints in QR-DQN paper),
            must be either (batch_size, n_quantiles), (batch_size, 1, n_quantiles), or (batch_size, n_critics, n_quantiles).
            (if None, calculating unit quantiles)
        :param sum_over_quantiles: if summing over the quantile dimension or not
        :param huber: If False, computes the pure quantile-regression loss
        :return: the loss
        """
        if dtd_est_locs.ndim != l_dtd_tgt.ndim:
            raise ValueError(
                f"Error: The dimension of current locations ({dtd_est_locs.ndim}) needs to match "
                f"the dimension of target locations ({l_dtd_tgt.ndim})."
            )
        if dtd_est_locs.shape[0] != l_dtd_tgt.shape[0]:
            raise ValueError(
                f"Error: The batch size of current locations ({dtd_est_locs.shape[0]}) needs to match "
                f"the batch size of target locations ({l_dtd_tgt.shape[0]})."
            )
        if dtd_est_locs.ndim not in (2, 3):
            raise ValueError(f"Error: The dimension of current locations ({dtd_est_locs.ndim}) needs to be either 2 or 3.")

        if self.order:
            dtd_est_locs = torch.sort(dtd_est_locs, descending=False, dim=-1)[0]

        if cum_prob is None:
            n_quantiles = dtd_est_locs.shape[-1]
            # Cumulative probabilities to calculate quantiles.
            cum_prob = (torch.arange(n_quantiles, device=dtd_est_locs.device, dtype=torch.float) + 0.5) / n_quantiles
            
            # make cum_prob broadcastable to (bs, n_atoms, n_targets)
            if dtd_est_locs.ndim == 2:
                cum_prob = cum_prob.view(1, -1, 1)
            elif dtd_est_locs.ndim == 3:
                cum_prob = cum_prob.view(1, 1, -1, 1)

        # QR
        # -> (bs, n_models, n_quantiles, n_target_locs)
        pairwise_delta = l_dtd_tgt.unsqueeze(-2) - dtd_est_locs.unsqueeze(-1)
        abs_pairwise_delta = torch.abs(pairwise_delta)
        if self.use_huber:
            huber_loss = torch.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5 , pairwise_delta ** 2 * 0.5)
        else: 
            huber_loss = abs_pairwise_delta
        
        # -> (bs, n_models, n_target_locs)
        is_ratio = p_dtd_tgt * l_dtd_tgt.shape[-1]

        # -> (bs, n_models, n_quantiles, n_target_locs)
        loss = is_ratio.unsqueeze(-2) * (torch.abs(cum_prob - (pairwise_delta.detach() < 0).float()) * huber_loss)
    
        if self.sum_over_quantiles:
            # -> (bs, n_models, n_target_locs)
            loss = loss.sum(dim=-2)
        else:
            loss = loss
        return loss
