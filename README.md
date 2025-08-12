# Projection-ensemble DQN
This is the official implementation of a distributional projection-ensemble DQN accompanying the paper "Diverse projection ensembles for distributional reinforcement learning" (ICLR 2024, [Link to arXiv](https://arxiv.org/abs/2306.07124) )

![](vizualization/pedqn_compressed.gif)


# Installation
The recommended way of installation for torch and cuda is via conda (tested for Python 3.8, Pytorch 1.13.1 and Cuda 11.6). The remaining packages can be installed via pip:
older gym version prequires setuptools and pip to use older versions.
```
cd projection-ensemble-dqn
conda create -n pedqn python=3.8
conda activate pedqn
conda install setuptools==65.5.0 pip==21.2.4 wheel==0.38.4 pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r req.txt
```

# VizDoom
Instructions on how to installing VizDoom can be found here: https://github.com/Farama-Foundation/ViZDoom. (Especially if a non-systemwide installation is required). 

For vizdoom environments, install vizdoomgym submodule and install per pip:
```
cd projection-ensemble-dqn
git submodule init
git submodule update --remote --init
cd envs/vizdoomgym/
pip install -e .
cd ..
```

# Running Experiments

Experiments can be run via the "train_\<algo\>.py" files. For example, a PEDQN experiment on vizdoom with 16 parallel environments can be run via the command

```
cd projection-ensemble-dqn

python train_pedqn.py --env_name=gym:VizdoomMyWayHome-v0 --batch_size=512 --parallel_envs=16 --train_count=1 --subproc_vecenv=True --decorr=False --burnin=10000 --init_beta=1.0 --final_beta=0.01 --gamma_i=0.99 --gamma=0.999 --beta_decay_frames=1_000_000 --prior_beta=3.0 --resolution=42 --frame_skip=1 --frame_stack=6 --stack_skip=1 --q_max=2 --q_min=-2 --episodic_ri=True --wandb_tag="PEDQN" --n_step=7 --lr=0.000625 --sync_dqn_target_every=8000 --record_every=10 --eps_decay_frames=50000
```

For logging, we use [wandb.ai](https://wandb.ai/site/) by default. 

# Hyperparameters 

The default hyperparameters used in our experiments can be found in the "argp_\<algo\>.py" files in the common folder. 

# Citing projection-ensemble DQN

To cite this work, please cite

```
@inproceedings{
zanger2024diverse,
title={Diverse Projection Ensembles for Distributional Reinforcement Learning},
author={Moritz Akiya Zanger and Wendelin Boehmer and Matthijs T. J. Spaan},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=qe49ybvvPs}
}
```
