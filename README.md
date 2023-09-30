# DMBP: Diffusion Model-Based Predictor for robust offline reinforcement learning against  state observation perturbations

This is the implementation of Decision Model-Based Predictor (DMBP) and the reproduced baseline algorithms (including [BCQ](https://arxiv.org/abs/1812.02900), [CQL](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html), [TD3+BC](https://proceedings.neurips.cc/paper/2021/hash/a8166da05c5a094f7dc03724b41886e5-Abstract.html), [RORL](https://arxiv.org/abs/2206.02829), and [Diffusion QL](https://arxiv.org/abs/2208.06193)).

## Introduction

A major challenge for the real-world application of offline RL stems from the robustness against state observation perturbations, *e.g.*, as a result of sensor errors or adversarial attacks. Unlike online robust RL, agents cannot be adversarially trained in the offline setting.

Our proposed Diffusion Model-Based Predictor (DMBP) is a new framework that recovers the actual states with conditional diffusion models for state-based RL tasks. Our derived non-Markovian training objective reduces the **error accumulations**, which are commonly observed for model-based methods in state-based RL tasks.

Follow is the visualization of the DMBP denoising effect on Mujoco Hopper-v2 with Gaussian noise (std of 0.1)

 <img src="/Hopper_medium_replay.gif" width = "700" height = "700" alt="DMBP_Visualization" align=center />  


## Requirement
Our experiment is performed on D4RL benchmark environments and datasets ([click here](https://sites.google.com/view/d4rl-anonymous/)).
Please install the Mujoco Version 2.1 
([click here](https://github.com/deepmind/mujoco/releases)) before getting start. See `requirements.txt` for detailed environment set up.  

## Training
### Baseline Algorithms Training
Before training DMBP, train a baseline offline RL algorithm at first:
```bash
python -m scripts.train_baseline --algo [ALGORITHM_NAME] --env_name [ENV_NAME] --dataset [DATASET_NAME]
```
### DMBP Training
DMBP utilizes the trajectory datasets for training. Download the datasets of the corresponding domain through
```bash
python -m scripts.download_datasets --domain [DOMAIN_NAME]
```
Then DMBP can be trained through:
```bash
python -m scripts.train_DMBP --task [TASK_NAME] --algo [ALGORITHM_NAME] --env_name [ENV_NAME] --dataset [DATASET_NAME]
```
where the previously trained baseline algorithms are only used for training-process evaluation.

## Evaluation
### Robustness against noised state observations
To evaluate the baseline algorithm against different attacks on state observations, run:
```bash
python -m evaluations.eval_baseline_noise --noise_type [ATTACK_METHOD] --algo [ALGORITHM_NAME] --env_name [ENV_NAME] --dataset [DATASET_NAME]
```
Then, the evaluation on the corresponding baseline algorithm strengthed by DMBP can be conducted through:
```bash
python -m evaluations.eval_DMBP_noise --noise_type [ATTACK_METHOD] --algo [ALGORITHM_NAME] --env_name [ENV_NAME] --dataset [DATASET_NAME]
```
### Robustness against incomplete state observations with unobserved dimensions


## Visualization
