# My Project with VERL Submodule

This repository includes [VERL (Volcengine Reinforcement Learning)](https://github.com/volcengine/verl) as a submodule.

## Setup

To clone this repository with its submodules:

```bash
git clone --recursive [your-repo-url]
```

Or if you've already cloned it:

```bash
git submodule update --init --recursive
```


## TODO
1. Implement the validation in ray_trainer
2. Transfer the old ray_trainer file from RAGEN to VAGEN
    - Implement the metric for wandb
3. Transfer the train.py from RAGEN to VAGEN
4. Change recorder to a logging class