# TD3-context Experimental Repo

PENG Zhenghao, LI Yunxinag

This is the internal repo for investigating the topics of meta-learning in Reinforcement Learning.

## Installation

Please follow the scripts to setup the environment.

```bash
git clone https://github.com/pengzhenghao/metaforce.git
cd metaforce
conda create -n metaforce python=3.7
conda activate metaforce
pip install -e .
```

Besides, you should also install pytorch following the [official tutorial](https://pytorch.org/get-started/locally/).

## Quick Start

The following script will run TD3-context experiment locally, which is useful for debugging.
```bash
python -m metaforce.run_experiment --context-mode add_both_transition
```

We also provide interface to call Ray as helper to organize large-scale experiments with grid-searching hyper-parameters:

```bash
python -m metaforce.run_batch_experiment --exp-name TEST --num-gpus 0 --num-seeds 3 --env ml10 --test-mode
```

## Best practice

Here is the best practice for launching experiment and developing.

###  Step 1: create independent directory for each experiment

```bash
cd metaforce/experiments
mkdir 0829_td3_context
cd 0829_td3_context
```

### Step 2: prepare your training script

```bash
# You are in .../metaforce/experiments/0829_td3_context now!
cp .../metaforce/experiments/example.py 0829_td3_context.py

# Modify the train script if necessary. You can also add those train script into git.
...
```

### Step 3: launch training script

```bash
nohup python 0829_td3_context.py --exp-name 0829_td3_context > 0829_td3_context.log 2>&1 &
```

### Step 4: observe the training progress

```bash
tail -n 100 -f 0829_td3_context.log
```

* [ ] TODO: add more document on how to plot the results.
