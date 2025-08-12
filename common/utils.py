import zlib
from copy import deepcopy
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple, Type,
                    Union)

import torch
from tqdm.auto import trange
import numpy as np
from common.env_wrappers import create_env


def prep_observation_for_qnet(tensor, use_amp):
    """ Tranfer the tensor the gpu and reshape it into (batch, frame_stack*channels, y, x) """
    assert len(tensor.shape) == 5, tensor.shape # (batch, frame_stack, y, x, channels)
    tensor = tensor.cuda().permute(0, 1, 4, 2, 3) # (batch, frame_stack, channels, y, x)
    # .cuda() needs to be before this ^ so that the tensor is made contiguous on the gpu
    tensor = tensor.reshape((tensor.shape[0], tensor.shape[1]*tensor.shape[2], *tensor.shape[3:]))

    return tensor.to(dtype=(torch.float16 if use_amp else torch.float32)) / 255

class LinearSchedule:
    """Set up a linear hyperparameter schedule (e.g. for dqn's epsilon parameter)"""

    def __init__(self, burnin: int, initial_value: float, final_value: float, decay_time: int):
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_time = decay_time
        self.burnin = burnin

    def __call__(self, frame: int) -> float:
        if frame < self.burnin:
            return self.initial_value
        else:
            frame = frame - self.burnin

        slope = (self.final_value - self.initial_value) / self.decay_time
        if self.final_value < self.initial_value:
            return max(slope * frame + self.initial_value, self.final_value)
        else:
            return min(slope * frame + self.initial_value, self.final_value)

class ExpDecaySchedule:
    """Set up an expontial decay hyperparameter schedule (e.g. for DLTV beta)"""

    def __init__(self, burnin: int, initial_value: float, decay_scale: float):
        self.initial_value = initial_value
        self.decay_scale = decay_scale
        self.burnin = burnin

    def __call__(self, frame: int) -> float:
        if frame < self.burnin:
            return self.initial_value
        else:
            frame = frame - self.burnin

        # +3 to keep values at 1 for first frame
        if self.decay_scale*(frame+3)<1.0:
            decay = 1
        else:
            decay = np.log(self.decay_scale*(frame+3))/(self.decay_scale*(frame+3))

        return self.initial_value * np.sqrt(decay)
    
def get_mean_ep_length(args):
    """Run a few iterations of the environment and estimate the mean episode length"""
    dc_args = deepcopy(args)
    dc_args.parallel_envs = 12
    dc_args.subproc_vecenv = True
    dc_env = create_env(dc_args)
    dc_env.reset()

    # Decorrelate envs
    ep_lengths = []
    for frame in trange(args.time_limit//4+100):
        _, _, _, infos = dc_env.step([dc_env.action_space.sample() for x in range(dc_args.parallel_envs)])
        for info, j in zip(infos, range(dc_args.parallel_envs)):
            if 'episode_metrics' in info.keys(): ep_lengths.append(info['episode_metrics']['length'])
    dc_env.close()
    mean_length = sum(ep_lengths)/len(ep_lengths)
    return mean_length

def env_seeding(user_seed, env_name):
    return user_seed + zlib.adler32(bytes(env_name, encoding='utf-8')) % 10000

class RunningMeanStdTh(torch.nn.Module):
    def __init__(self, epsilon: float = 1e-8, shape: Tuple[int, ...] = (), device: Optional[str] = "cpu", fixed_size: bool = -1):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        :param fixed_size: if set to a positive integer, the running mean and std will be calculated only over the last fixed_size number of elements
        """
        super(RunningMeanStdTh, self).__init__()

        self.epsilon = epsilon
        self.run_mean = torch.nn.Parameter(torch.zeros(size = shape, device=device, requires_grad=False))
        self.run_var = torch.nn.Parameter(torch.ones(size = shape, device=device, requires_grad=False))
        self.run_count = torch.nn.Parameter(torch.tensor(self.epsilon, device=device, requires_grad=False))
        self.run_min = torch.nn.Parameter(torch.tensor(1e9, device=device, requires_grad=False))
        self.fixed_size = fixed_size
        if self.fixed_size > 0:
            self.running_buffer = torch.nn.Parameter(torch.zeros(size = (fixed_size,) + shape, device=device, requires_grad=False))
            self.running_index = 0
            self.full = False
        self.discount = 0.999 # e.g.: 0.99 makes makes an effective dataset size of 1e2

    def copy(self) -> "RunningMeanStdTh":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStdTh(shape=self.run_mean.shape, fixed_size=self.fixed_size)
        new_object.run_mean = self.run_mean.copy()
        new_object.run_var = self.run_var.copy()
        new_object.run_count = float(self.run_count)
        if self.fixed_size>0:
            new_object.running_buffer = self.running_buffer.copy()
            new_object.running_index = self.running_index

        return new_object

    def forward(self, x: torch.Tensor, adjustmean: bool = True, fractionalmean:float=1.0, zeromin: bool = False) -> torch.Tensor:
        if adjustmean:
            x_norm = (x - fractionalmean * self.run_mean) / torch.sqrt(self.run_var + self.epsilon)
        else:
            x_norm = (x) / torch.sqrt(self.run_var + self.epsilon)
        return x_norm

    def combine(self, other: "RunningMeanStdTh") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine witorch.
        """
        if self.fixed_size>0 and other.fixed_size>0:
            self.update(other.running_buffer[:other.running_index])
        elif self.fixed_size>0 and other.fixed_size<0:
            raise ValueError("Cannot combine fixed size with infinite size")
        else:
            self.update_from_moments(other.run_mean, other.run_var, other.run_count)

    def update(self, tensor: torch.Tensor) -> None:
        if self.fixed_size>0:
            batch_count = tensor.shape[0]
            if self.running_index+batch_count>=self.fixed_size:
                self.full = True
                n_fill_elements = self.fixed_size - self.running_index
                self.running_buffer[self.running_index:] = tensor[:n_fill_elements]
                self.running_buffer[:batch_count-n_fill_elements] = tensor[n_fill_elements:]
                self.running_index = batch_count-n_fill_elements
            else:
                self.running_buffer[self.running_index:self.running_index+batch_count] = tensor
                self.running_index += batch_count
            
            if self.full:
                self.run_mean.data = self.running_buffer.mean(dim=0)
                self.run_var.data = self.running_buffer.var(dim=0)
                self.run_count.data = torch.tensor(self.fixed_size, device=self.run_count.device, requires_grad=False, dtype=torch.float32)
                self.run_min.data = self.running_buffer.min(dim=0)[0]
            else:
                self.run_mean.data = self.running_buffer[:self.running_index].mean(dim=0)
                self.run_var.data = self.running_buffer[:self.running_index].var(dim=0)
                self.run_count.data = torch.tensor(self.running_index, device=self.run_count.device, requires_grad=False, dtype=torch.float32)
                self.run_min.data = self.running_buffer[:self.running_index].min(dim=0)[0]
        else:
            batch_mean = torch.mean(tensor, dim=0)
            batch_var = torch.var(tensor, dim=0, unbiased=False) #biased to avoid nan
            batch_count = tensor.shape[0]
            batch_min = torch.min(tensor)
            self.update_from_moments(batch_mean, batch_var, batch_min, batch_count)

    def update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_min: torch.Tensor, batch_count: Union[int, float]) -> None:
        self.run_count.data = self.run_count.data * self.discount
        
        delta = batch_mean - self.run_mean
        tot_count = self.run_count + batch_count

        new_mean = self.run_mean + delta * batch_count / tot_count
        m_a = self.run_var * self.run_count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.run_count * batch_count / (tot_count)
        new_var = m_2 / tot_count

        new_min = min(self.run_min/self.discount, batch_min)
        
        self.run_mean.data = new_mean
        self.run_var.data = new_var
        self.run_count.data = tot_count
        self.run_min.data = new_min