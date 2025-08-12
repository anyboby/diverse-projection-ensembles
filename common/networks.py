"""
This file defines all the neural network architectures available to use.
"""
from functools import partial
from math import sqrt

import torch
from torch import nn as nn, Tensor
from torch.nn import init
import torch.nn.functional as F

class FactorizedNoisyLinear(nn.Module):
    """ The factorized Gaussian noise layer for noisy-nets dqn. """
    def __init__(self, in_features: int, out_features: int, sigma_0: float) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0

        # weight: w = \mu^w + \sigma^w . \epsilon^w
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        # bias: b = \mu^b + \sigma^b . \epsilon^b
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # initialization is similar to Kaiming uniform (He. initialization) with fan_mode=fan_in
        scale = 1 / sqrt(self.in_features)

        init.uniform_(self.weight_mu, -scale, scale)
        init.uniform_(self.bias_mu, -scale, scale)

        init.constant_(self.weight_sigma, self.sigma_0 * scale)
        init.constant_(self.bias_sigma, self.sigma_0 * scale)

    @torch.no_grad()
    def _get_noise(self, size: int) -> Tensor:
        noise = torch.randn(size, device=self.weight_mu.device)
        # f(x) = sgn(x)sqrt(|x|)
        return noise.sign().mul_(noise.abs().sqrt_())

    @torch.no_grad()
    def reset_noise(self) -> None:
        # like in eq 10 and 11 of the paper
        epsilon_in = self._get_noise(self.in_features)
        epsilon_out = self._get_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @torch.no_grad()
    def disable_noise(self) -> None:
        self.weight_epsilon[:] = 0
        self.bias_epsilon[:] = 0

    def forward(self, input: Tensor) -> Tensor:
        # y = wx + d, where
        # w = \mu^w + \sigma^w * \epsilon^w
        # b = \mu^b + \sigma^b * \epsilon^b
        return F.linear(input,
                        self.weight_mu + self.weight_sigma*self.weight_epsilon,
                        self.bias_mu + self.bias_sigma*self.bias_epsilon)

class Dueling(nn.Module):
    """ The dueling branch used in all nets that use dueling-dqn. """
    def __init__(self, value_branch, advantage_branch):
        super().__init__()
        self.flatten = nn.Flatten()
        self.value_branch = value_branch
        self.advantage_branch = advantage_branch

    def forward(self, x, advantages_only=False):
        x = self.flatten(x)
        advantages = self.advantage_branch(x)
        if advantages_only:
            return advantages

        value = self.value_branch(x)
        return value + (advantages - torch.mean(advantages, dim=1, keepdim=True))


class DuelingAlt(nn.Module):
    """ The dueling branch used in all nets that use dueling-dqn. """
    def __init__(self, l1, l2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            l1,
            nn.ReLU(),
            l2
        )

    def forward(self, x, advantages_only=False):
        res = self.main(x)
        advantages = res[:, 1:]
        value = res[:, 0:1]
        return value + (advantages - torch.mean(advantages, dim=1, keepdim=True))
    
class EnsembleModule(nn.Module):
    def __init__(self, models, shared_body):
        super(EnsembleModule, self).__init__()
        self.models = nn.ModuleList(models)
        self.shared_body = shared_body
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)

        if self.shared_body is not None:
            x = self.shared_body(x)

        forwards = []
        for m in self.models:
            x_n = m(x)
            forwards.append(x_n.unsqueeze(1))

        x = torch.cat(forwards, dim=1)
        
        return x

class NatureCNN(nn.Module):
    """
    This is the CNN that was introduced in Mnih et al. (2013) and then used in a lot of later work such as
    Mnih et al. (2015) and the Rainbow paper. This implementation only works with a frame resolution of 84x84.
    """
    def __init__(self, depth, actions, linear_layer):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            linear_layer(3136, 512),
            nn.ReLU(),
            linear_layer(512, actions),
        )

    def forward(self, x, advantages_only=None):
        return self.main(x)

class DuelingNatureCNN(nn.Module):
    """
    Implementation of the dueling architecture introduced in Wang et al. (2015).
    This implementation only works with a frame resolution of 84x84.
    """
    def __init__(self, depth, actions, linear_layer):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.dueling = Dueling(
                nn.Sequential(linear_layer(3136, 512),
                              nn.ReLU(),
                              linear_layer(512, 1)),
                nn.Sequential(linear_layer(3136, 512),
                              nn.ReLU(),
                              linear_layer(512, actions))
            )

    def forward(self, x, advantages_only=False):
        f = self.main(x)
        return self.dueling(f, advantages_only=advantages_only)


class ImpalaCNNSmall(nn.Module):
    """
    Implementation of the small variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, depth, actions, linear_layer):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        self.pool = torch.nn.AdaptiveMaxPool2d((6, 6))

        self.dueling = Dueling(
                nn.Sequential(linear_layer(1152, 256),
                              nn.ReLU(),
                              linear_layer(256, 1)),
                nn.Sequential(linear_layer(1152, 256),
                              nn.ReLU(),
                              linear_layer(256, actions))
            )

    def forward(self, x, advantages_only=False):
        f = self.main(x)
        f = self.pool(f)
        return self.dueling(f, advantages_only=advantages_only)


class ImpalaCNNResidual(nn.Module):
    """
    Simple residual block used in the large IMPALA CNN.
    """
    def __init__(self, depth, norm_func):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv_0 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))
        self.conv_1 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x_ = self.conv_0(self.relu(x))
        x_ = self.conv_1(self.relu(x_))
        return x+x_

class ImpalaCNNBlock(nn.Module):
    """
    Three of these blocks are used in the large IMPALA CNN.
    """
    def __init__(self, depth_in, depth_out, norm_func):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=depth_in, out_channels=depth_out, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)
        self.residual_0 = ImpalaCNNResidual(depth_out, norm_func=norm_func)
        self.residual_1 = ImpalaCNNResidual(depth_out, norm_func=norm_func)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.residual_0(x)
        x = self.residual_1(x)
        return x


class ImpalaCNNLarge(nn.Module):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, in_depth, actions, linear_layer, model_size=1, spectral_norm=False):
        super().__init__()

        def identity(p): return p

        norm_func = torch.nn.utils.spectral_norm if (spectral_norm == 'all') else identity
        norm_func_last = torch.nn.utils.spectral_norm if (spectral_norm == 'last' or spectral_norm == 'all') else identity

        self.main = nn.Sequential(
            ImpalaCNNBlock(in_depth, 16*model_size, norm_func=norm_func),
            ImpalaCNNBlock(16*model_size, 32*model_size, norm_func=norm_func),
            ImpalaCNNBlock(32*model_size, 32*model_size, norm_func=norm_func_last),
            nn.ReLU()
        )

        self.pool = torch.nn.AdaptiveMaxPool2d((8, 8))

        self.dueling = Dueling(
            nn.Sequential(linear_layer(2048*model_size, 256),
                          nn.ReLU(),
                          linear_layer(256, 1)),
            nn.Sequential(linear_layer(2048*model_size, 256),
                          nn.ReLU(),
                          linear_layer(256, actions))
        )

    def forward(self, x, advantages_only=False):
        f = self.main(x)
        f = self.pool(f)
        return self.dueling(f, advantages_only=advantages_only)
    
class ImpalaCNNBDQN(nn.Module):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018) adapted for BDQN.
    """
    def __init__(self, in_depth, actions, linear_layer, model_size=1, spectral_norm=False, prior_beta=0, ensemble_size=1):
        super().__init__()

        def identity(p): return p

        norm_func = torch.nn.utils.spectral_norm if (spectral_norm == 'all') else identity
        norm_func_last = torch.nn.utils.spectral_norm if (spectral_norm == 'last' or spectral_norm == 'all') else identity

        self.prior_beta=prior_beta
        self.actions = actions
        self.ensemble_size = ensemble_size

        self.main = nn.Sequential(
            ImpalaCNNBlock(in_depth, 16*model_size, norm_func=norm_func),
            ImpalaCNNBlock(16*model_size, 32*model_size, norm_func=norm_func),
            ImpalaCNNBlock(32*model_size, 32*model_size, norm_func=norm_func_last),
            nn.ReLU()
        )

        self.pool = torch.nn.AdaptiveMaxPool2d((8, 8))

        nn_ens = [nn.Sequential(linear_layer(2048*model_size, 256),
                                        nn.ReLU(),
                                        linear_layer(256, actions)) for _ in range(ensemble_size)]

        self.seq_ensemble = EnsembleModule(nn_ens, shared_body=None)

        if self.prior_beta>0:
            prior_fs = [nn.Sequential(linear_layer(2048*model_size, 256),
                                        nn.ReLU(),
                                        linear_layer(256, actions)
                ) for _ in range(ensemble_size)]
            
            self.prior_ens = EnsembleModule(prior_fs,
                                        shared_body=None
                )

    def forward(self, x):
        x = self.main(x)
        x = self.pool(x)
        x_ = self.seq_ensemble(x)
        if self.prior_beta>0:
            with torch.no_grad():
                x_pr = self.prior_beta * self.prior_ens(x)
            x_ = x_ + x_pr
        x_ = x_.view(-1, x_.shape[1], self.actions)
        return x_

class ImpalaCNNIDS(nn.Module):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, in_depth, actions, linear_layer, model_size=1, prior_beta=3.0, ensemble_size=1, n_heads=51, spectral_norm=False):
        super().__init__()

        def identity(p): return p

        norm_func = torch.nn.utils.spectral_norm if (spectral_norm == 'all') else identity
        norm_func_last = torch.nn.utils.spectral_norm if (spectral_norm == 'last' or spectral_norm == 'all') else identity
        
        self.actions = actions
        self.n_heads = n_heads
        self.prior_beta = prior_beta

        self.main = nn.Sequential(
            ImpalaCNNBlock(in_depth, 16*model_size, norm_func=norm_func),
            ImpalaCNNBlock(16*model_size, 32*model_size, norm_func=norm_func),
            ImpalaCNNBlock(32*model_size, 32*model_size, norm_func=norm_func_last),
            nn.ReLU()
        )

        self.pool = torch.nn.AdaptiveMaxPool2d((8, 8))

        nn_ens = [nn.Sequential(linear_layer(2048*model_size, 256),
                                        nn.ReLU(),
                                        linear_layer(256, actions)) for _ in range(ensemble_size)]

        self.seq_ensemble = EnsembleModule(nn_ens, shared_body=None)

        if self.prior_beta>0:
            prior_fs = [nn.Sequential(linear_layer(2048*model_size, 256),
                                        nn.ReLU(),
                                        linear_layer(256, actions)
                ) for _ in range(ensemble_size)]
            
            self.prior_ens = EnsembleModule(prior_fs,
                                        shared_body=None
                )
            
        self.catlinear = EnsembleModule([nn.Sequential(linear_layer(2048*model_size, 256),
                                        nn.ReLU(),
                                        linear_layer(256, actions*n_heads)),], shared_body=None
        )
        
    def forward(self, x):
        x = self.main(x)
        x = self.pool(x)

        x_ = self.seq_ensemble(x)
        if self.prior_beta>0:
            with torch.no_grad():
                x_pr = self.prior_beta * self.prior_ens(x)
            x_ = x_ + x_pr
        x_ = x_.view(-1, x_.shape[1], self.actions)

        x_d = self.catlinear(x)
        x_d = x_d.view(-1, self.n_heads, self.actions)
        return x_, x_d


class ImpalaCNNDLTV(nn.Module):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, in_depth, actions, linear_layer, model_size=1, prior_beta=3.0, n_heads=51, spectral_norm=False):
        super().__init__()

        def identity(p): return p

        norm_func = torch.nn.utils.spectral_norm if (spectral_norm == 'all') else identity
        norm_func_last = torch.nn.utils.spectral_norm if (spectral_norm == 'last' or spectral_norm == 'all') else identity
        
        self.actions = actions
        self.n_heads = n_heads
        self.prior_beta = prior_beta

        self.main = nn.Sequential(
            ImpalaCNNBlock(in_depth, 16*model_size, norm_func=norm_func),
            ImpalaCNNBlock(16*model_size, 32*model_size, norm_func=norm_func),
            ImpalaCNNBlock(32*model_size, 32*model_size, norm_func=norm_func_last),
            nn.ReLU()
        )

        self.pool = torch.nn.AdaptiveMaxPool2d((8, 8))


        self.quantlinear = EnsembleModule([nn.Sequential(linear_layer(2048*model_size, 256),
                                        nn.ReLU(),
                                        linear_layer(256, actions*n_heads)),], shared_body=None
        )

        if self.prior_beta>0:
            self.prior_quant = EnsembleModule([nn.Sequential(linear_layer(2048*model_size, 256),
                                        nn.ReLU(),
                                        linear_layer(256, actions*n_heads)),], shared_body=None
            )
        
    def forward(self, x):
        x = self.main(x)
        x = self.pool(x)

        x_ = self.quantlinear(x)
        if self.prior_beta>0:
            with torch.no_grad():
                x_pr = self.prior_beta * self.prior_quant(x)
            x_ = x_ + x_pr
        
        x_ = x_.view(-1, self.n_heads, self.actions)
        return x_

class ImpalaCNNSIMD_QNET(nn.Module):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, in_depth, actions, linear_layer, model_size=1, prior_beta=0, spectral_norm=False):
        super().__init__()

        def identity(p): return p

        norm_func = torch.nn.utils.spectral_norm if (spectral_norm == 'all') else identity
        norm_func_last = torch.nn.utils.spectral_norm if (spectral_norm == 'last' or spectral_norm == 'all') else identity
        
        self.actions = actions
        self.prior_beta = prior_beta

        self.main = nn.Sequential(
            ImpalaCNNBlock(in_depth, 16*model_size, norm_func=norm_func),
            ImpalaCNNBlock(16*model_size, 32*model_size, norm_func=norm_func),
            ImpalaCNNBlock(32*model_size, 32*model_size, norm_func=norm_func_last),
            nn.ReLU()
        )

        self.pool = torch.nn.AdaptiveMaxPool2d((8, 8))
        
        self.q_linear = nn.Sequential(linear_layer(2048*model_size, 256),
                                        nn.ReLU(),
                                        linear_layer(256, actions)
        )
        self.u_linear = nn.Sequential(linear_layer(2048*model_size, 256),
                                        nn.ReLU(),
                                        linear_layer(256, actions)
        )
        self.linear_ens = EnsembleModule([self.q_linear, self.u_linear],
                                        shared_body=None
        )

        if self.prior_beta>0:
            prior_ens = [ nn.Sequential(linear_layer(2048*model_size, 256),
                                        nn.ReLU(),
                                        linear_layer(256, actions)
                ) for _ in range(2) ]
            self.prior_ens = EnsembleModule(prior_ens, shared_body=None)

    def forward(self, x):
        x = self.main(x)
        x = self.pool(x)
        x_ = self.linear_ens(x)
        if self.prior_beta>0:
            with torch.no_grad():
                x_pr = self.prior_beta * self.prior_ens(x)
            x_ = x_ + x_pr
        x_ = x_.view(-1, x_.shape[1], self.actions)
        return x_


class ImpalaCNNSIMDC51(nn.Module):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, in_depth, actions, linear_layer, n_heads, model_size=1, prior_beta=0, spectral_norm=False):
        super().__init__()

        def identity(p): return p

        norm_func = torch.nn.utils.spectral_norm if (spectral_norm == 'all') else identity
        norm_func_last = torch.nn.utils.spectral_norm if (spectral_norm == 'last' or spectral_norm == 'all') else identity
        
        self.actions = actions
        self.n_heads = n_heads
        self.prior_beta = prior_beta

        self.main = nn.Sequential(
            ImpalaCNNBlock(in_depth, 16*model_size, norm_func=norm_func),
            ImpalaCNNBlock(16*model_size, 32*model_size, norm_func=norm_func),
            ImpalaCNNBlock(32*model_size, 32*model_size, norm_func=norm_func_last),
            nn.ReLU()
        )

        self.pool = torch.nn.AdaptiveMaxPool2d((8, 8))
        
        self.c51_linear = nn.Sequential(linear_layer(2048*model_size, 256),
                                        nn.ReLU(),
                                        linear_layer(256, actions*n_heads)
        )
        self.u_c51_linear = nn.Sequential(linear_layer(2048*model_size, 256),
                                        nn.ReLU(),
                                        linear_layer(256, actions*n_heads)
        )

        self.linear_ens = EnsembleModule([self.c51_linear, self.u_c51_linear],
                                        shared_body=None
        )

        if self.prior_beta>0:
            raise NotImplementedError

    def forward(self, x):
        x = self.main(x)
        x = self.pool(x)
        x_ = self.linear_ens(x)
        if self.prior_beta>0:
            with torch.no_grad():
                x_pr = self.prior_beta * self.prior_ens(x)
            x_ = x_ + x_pr
        x_ = x_.view(-1, 2, self.n_heads, self.actions)
        return x_


class ImpalaCNNSIMD(nn.Module):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, in_depth, actions, linear_layer, n_linear_layers, n_heads, model_size=1, prior_beta=0, spectral_norm=False):
        super().__init__()

        def identity(p): return p

        norm_func = torch.nn.utils.spectral_norm if (spectral_norm == 'all') else identity
        norm_func_last = torch.nn.utils.spectral_norm if (spectral_norm == 'last' or spectral_norm == 'all') else identity
        
        self.actions = actions
        self.n_heads = n_heads
        self.prior_beta = prior_beta

        self.main = nn.Sequential(
            ImpalaCNNBlock(in_depth, 16*model_size, norm_func=norm_func),
            ImpalaCNNBlock(16*model_size, 32*model_size, norm_func=norm_func),
            ImpalaCNNBlock(32*model_size, 32*model_size, norm_func=norm_func_last),
            nn.ReLU()
        )

        self.pool = torch.nn.AdaptiveMaxPool2d((8, 8))
        
        linearmodules = [linear_layer(2048*model_size, 256), nn.ReLU(),]
        for _ in range(n_linear_layers-1):
            linearmodules.extend([linear_layer(256, 256), nn.ReLU()])
        linearmodules.extend([linear_layer(256, actions*n_heads)])
        self.linear = nn.Sequential(*linearmodules)
        self.linear_ens = EnsembleModule([self.linear],
                                        shared_body=None
        )

        if self.prior_beta>0:
            # only applies to quantile network
            priormodules = [linear_layer(2048*model_size, 256), nn.ReLU(),]
            for _ in range(n_linear_layers-1):
                priormodules.extend([linear_layer(256, 256), nn.ReLU()])
            priormodules.extend([linear_layer(256, actions*n_heads)])
            prior_ens = nn.Sequential(*priormodules)
            self.prior_ens = EnsembleModule([prior_ens], shared_body=None)

    def forward(self, x, normalize=True):
        x = self.main(x)
        x = self.pool(x)
        x_ = self.linear_ens(x)
        if self.prior_beta>0:
            with torch.no_grad():
                x_pr = self.prior_beta * self.prior_ens(x)
            x_ = x_ + x_pr
        x_ = x_.view(-1, self.n_heads, self.actions)
        if normalize:
            x_ = torch.nn.functional.normalize(x_, p=2, dim=1, eps=1e-12)
        return x_


class ImpalaCNNPEDQN(nn.Module):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, in_depth, actions, linear_layer, n_heads, model_size=1, prior_beta=0, spectral_norm=False):
        super().__init__()

        def identity(p): return p

        norm_func = torch.nn.utils.spectral_norm if (spectral_norm == 'all') else identity
        norm_func_last = torch.nn.utils.spectral_norm if (spectral_norm == 'last' or spectral_norm == 'all') else identity
        
        self.actions = actions
        self.n_heads = n_heads
        self.prior_beta = prior_beta

        self.main = nn.Sequential(
            ImpalaCNNBlock(in_depth, 16*model_size, norm_func=norm_func),
            ImpalaCNNBlock(16*model_size, 32*model_size, norm_func=norm_func),
            ImpalaCNNBlock(32*model_size, 32*model_size, norm_func=norm_func_last),
            nn.ReLU()
        )

        self.pool = torch.nn.AdaptiveMaxPool2d((8, 8))
        
        self.catlinear = nn.Sequential(linear_layer(2048*model_size, 256),
                                        nn.ReLU(),
                                        linear_layer(256, actions*n_heads)
        )
        self.quantlinear = nn.Sequential(linear_layer(2048*model_size, 256),
                                        nn.ReLU(),
                                        linear_layer(256, actions*n_heads)
        )
        self.u_catlinear = nn.Sequential(linear_layer(2048*model_size, 256),
                                        nn.ReLU(),
                                        linear_layer(256, actions*n_heads)
        )
        self.u_quantlinear = nn.Sequential(linear_layer(2048*model_size, 256),
                                        nn.ReLU(),
                                        linear_layer(256, actions*n_heads)
        )

        self.linear_ens = EnsembleModule([self.catlinear, self.quantlinear, self.u_catlinear, self.u_quantlinear],
                                        shared_body=None
        )

        if self.prior_beta>0:
            # only applies to quantile network
            prior_ens = [nn.Sequential(linear_layer(2048*model_size, 256),
                                        nn.ReLU(),
                                        linear_layer(256, actions*n_heads)
                )]
            self.prior_ens = EnsembleModule(prior_ens, shared_body=None)

    def forward(self, x):
        x = self.main(x)
        x = self.pool(x)
        x_ = self.linear_ens(x)
        if self.prior_beta>0:
            with torch.no_grad():
                x_pr = self.prior_beta * self.prior_ens(x)[:,0]
            x_[:,1] = x_[:,1] + x_pr
        x_ = x_.view(-1, x_.shape[1], self.n_heads, self.actions)
        return x_


def get_model(model_str, spectral_norm):
    if model_str == 'nature': return NatureCNN
    elif model_str == 'dueling': return DuelingNatureCNN
    elif model_str == 'impala_small': return ImpalaCNNSmall
    elif model_str.startswith('impala_large:'):
        return partial(ImpalaCNNLarge, model_size=int(model_str[13:]), spectral_norm=spectral_norm)
    elif model_str.startswith('impala_pedqn:'):
        return partial(ImpalaCNNPEDQN, model_size=int(model_str[13:]), spectral_norm=spectral_norm)
    elif model_str.startswith('impala_simd_qnet:'):
        return partial(ImpalaCNNSIMD_QNET, model_size=int(model_str[17:]), spectral_norm=spectral_norm)    
    elif model_str.startswith('impala_simd_c51:'):
        return partial(ImpalaCNNSIMDC51, model_size=int(model_str[16:]), spectral_norm=spectral_norm)    
    elif model_str.startswith('impala_simd:'):
        return partial(ImpalaCNNSIMD, model_size=int(model_str[12:]), spectral_norm=spectral_norm)    
    elif model_str.startswith('impala_bdqn:'):
        return partial(ImpalaCNNBDQN, model_size=int(model_str[12:]), spectral_norm=spectral_norm)
    elif model_str.startswith('impala_ids:'):
        return partial(ImpalaCNNIDS, model_size=int(model_str[11:]), spectral_norm=spectral_norm)    
    elif model_str.startswith('impala_dltv:'):
        return partial(ImpalaCNNDLTV, model_size=int(model_str[12:]), spectral_norm=spectral_norm)    