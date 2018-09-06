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
written to an output file in the `Checkpoints` folder.


