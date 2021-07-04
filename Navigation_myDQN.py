#%%
from unityagents import UnityEnvironment
import numpy as np
from dqn_agent import Agent
from collections import deque
import matplotlib.pyplot as plt
import torch

'''
we will start the environment!  Before running the code cell below, change the `file_name` parameter to match the location of the Unity environment that you downloaded.

- **Mac**: `"path/to/Banana.app"`
- **Windows** (x86): `"path/to/Banana_Windows_x86/Banana.exe"`
- **Windows** (x86_64): `"path/to/Banana_Windows_x86_64/Banana.exe"`
- **Linux** (x86): `"path/to/Banana_Linux/Banana.x86"`
- **Linux** (x86_64): `"path/to/Banana_Linux/Banana.x86_64"`
- **Linux** (x86, headless): `"path/to/Banana_Linux_NoVis/Banana.x86"`
- **Linux** (x86_64, headless): `"path/to/Banana_Linux_NoVis/Banana.x86_64"`

For instance, if you are using a Mac, then you downloaded `Banana.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
env = UnityEnvironment(file_name="Banana.app")
'''
env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86")

########################################## setting ##########################################
'''
if you want to see agent's action slowly to see the result of 100 consecutive rewards,
then set mode = 'slow'
else if fastly, mode = 'fast'
''' 
# state = 'Train'
state = 'Test'

# mode = 'slow'
mode = 'fast'
########################################## setting ##########################################

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

agent = Agent(state_size=37, action_size=4, seed=0)

if state == 'Train':
    scores = agent.train(env, n_episodes=1800, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995)
else: # Test 
    agent.qnetwork_local.load_state_dict(torch.load('saved_DQN_agent.pth'))
    print('====================================')
    print('Sucessfully loaded from {}'.format('saved_DQN_agent.pth'))
    print('====================================')

    score_test = []                                  
    for test_episode in range(1, 100+1):
        train_mode = True if mode == 'fast' else False
        env_info = env.reset(train_mode=train_mode)[brain_name]      # reset the environment
        state = env_info.vector_observations[0]                      # get the current state
        score_temp = 0.                                              # initialize the score
        while True:
            action = agent.act(state, 0.)                            # select an action
            env_info = env.step(action)[brain_name]                  # send the action to the environment
            next_state = env_info.vector_observations[0]             # get the next state
            reward = env_info.rewards[0]                             # get the reward
            done = env_info.local_done[0]                            # see if episode has finished
            score_temp += reward                                     # update the score
            state = next_state                                       # roll over the state to next time step
            if done:                                                 # exit loop if episode finished
                score_test.append(score_temp)
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(test_episode, np.mean(score_test)), end="")
                score_temp = 0.
                break
        
    print("\nFinal Score: {}".format(np.mean(score_test)))
    env.close()


# %%
