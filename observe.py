from unityagents import UnityEnvironment
import random
import torch
import numpy as np
from collections import deque
import platform
from agents import Agent

# determine OS
platform = platform.system()
if platform == "Darwin":
    env_path = "Banana_Mac/Banana.app"
elif platform == "Windows":
    env_path = "Banana_Windows_x86_64/Banana.exe"
else:
    print("Unknown OS. Falling back to Windows environment binaries.")
    env_path = "Banana_Windows_x86_64/Banana.exe"

# create Unity environment
env = UnityEnvironment(file_name=env_path)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# get action and state vector dimensions
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

# create agent and load trained network
agent = Agent(state_size=state_size, action_size=action_size, seed=0)
agent.qnetwork_local.load_state_dict(torch.load('checkpoints/checkpoint.pth'))

env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = int(agent.act(state, 0))              # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    print("Score: {}".format(score), end = '\r')
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break
    
print("Score: {}".format(score))