import gym
import random
import torch
import numpy as np
from collections import deque
from Agent import Agent
import matplotlib.pyplot as plt


def watch_untrained():
    # watch an untrained agent
    state = env.reset()
    for j in range(200):
        action = agent.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            break 
            
    env.close()

def watch_trained():
    # load the weights from file
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    for i in range(3):
        state = env.reset()
        for j in range(400):
            action = agent.act(state)
            env.render()
            state, reward, done, _ = env.step(action)
            if done:
                break 
                
    env.close()


env = gym.make('LunarLander-v2')
env.seed(1005)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

agent = Agent(state_size=8, action_size=4, seed=0)

watch_trained()