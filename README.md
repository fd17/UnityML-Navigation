# UnityML-Navigation Project

This project is aimed at solving a provided Unity Machine Learning Agents Toolkit challenge. In this challenge, an
agent must learn to navigate a rectangular environment in order to collect fresh (yellow) bananas, while avoiding rotten
(blue) bananas. This solution uses Value based Reinforcement Learning with Neural Networks to approximate the Q-function (aka Deep Q-Learning).

## Requirements

* Windows (64 bit) or Mac OS
* [Python 3.6](https://www.python.org/downloads/release/python-366/)
* [Unity ML-Agents Toolkit](https://www.python.org/downloads/release/python-366/)
* [Pytorch](https://pytorch.org/)
* [Matplotlib](https://matplotlib.org/) (only for plotting training scores)
* [Jupyter](http://jupyter.org/) (only for the notebook implementation)

## Installation
Recommended way of installing the dependencies is via [Anaconda](https://www.anaconda.com/download/). To create a new Python 3.6 environment run

`conda create --name myenv python=3.6`

Activate the environment with

`conda activate myenv`

[Click here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) for instructions on how to install the Unity ML-Agents Toolkit.

Visit [pytorch.org](https://pytorch.org/) for instructions on installing pytorch.

Install matplotlib with

`conda install -c conda-forge matplotlib`

Jupyter should come installed with Anaconda. If not, [click here](http://jupyter.org/install) for instructions on how to install Jupyter.


## Getting started
There are two ways to run the project. 

* 1: Use Jupyter to run the provided `Navigation.ipynb` notebook. The notebook
contains all the necessary steps to setup the environment, train an agent and to observe a trained agent. 
* 2: Directly run the necessary python files in a console window
  * `train.py` to train an agent
  * `observe.py` to observe a trained agent
  
  Example usage: `python observe.py`

Both ways automatically detect the current operating system (Windows or Mac) and select the necessary Unity environment binaries
for the training environment.

## Environment
The environment is a large square world with a 37 dimensional state space. This space includes the agent's
velocity as well as a ray-based perception of nearby objects.
There are are four possible actions at each step:

| index   | action        |
|---------|---------------|
| 0       | move forward  |
| 1       | move backward |
| 2       | turn left     |
| 3       | turn right    |

## Agent
The Reinforcement Learning agent uses [Deep-Q-Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) with fixed Q-targets. The Q-function is approximated using fully connected
feed forward neural networks, with the inputs the size of the state space and a linear output layer of the same dimension as the action space.
The optimization process makes use of PyTorch's implementation of Adam. The following hyperparameters are used for the agent:

| parameter   | value        |  description |
|---------|---------------|-------------|
| buffer_size | 100000 | size of replay buffer |
| batch_size       | 64 | size of training batch |
| gamma       | 0.99     | discount factor |
| LR       | 0.0005    | learning rate |
|update_every| 4 | network training interval |

## Training
During training the agent is exposed to a maximum of 1000 steps per episode. At the end of each episode, the total
reward is stored in an array that contains the last 100 episode rewards. When the average reward of this array reaches
a value of 13.0, the environment is considered solved and the training terminates. The latest network parameters are then
written to an output file in the `Checkpoints` folder. Training uses the following parameters:

| parameter   | value    |  description |
|---------|---------------|-------------|
| n_episodes| 2000 | maximum number of episodes |
| max_t       | 1000 | maximum steps per episode|
| eps_start   | 0.99     | starting value for epsilon |
| eps_end     | 0.001    | minimum epsilon |
|eps_decay| 0.01 | epsilon decay per episode |

With these settings, the agent should learn to solve the environment 
within arrpoximately 500 episodes.

## Observation
This footage shows a fully trained agent acting in the environment:

<img src="https://github.com/fd17/UnityML-Navigation/blob/master/solved_showcase.gif" width="480" height="270" />

