# TD3-context Experimental Repo

PENG Zhenghao, LI Yunxinag

This is the internal repo for investigating the topics of meta-learning in Reinforcement Learning.

## Installation

Please follow the scripts to setup the environment.

```bash
conda create -n context python=3.7
conda activate context
pip install -r requirements.txt
```

Besides, you should also install pytorch following the [official tutorial](https://pytorch.org/get-started/locally/).

## Quick Start

```python
python run_experiment.py --env-name MountainCarContinuous-v0 --learn_start 2000 --context-mode "random" --batch_size 64
python run_experiment.py --env-name MountainCarContinuous-v0 --learn_start 2000 --context-mode "disable" --batch_size 64
```