from ray.tune.logger import pretty_print

from env import generate_and_save_maze

import numpy as np

#Parameters for the training
size = 17 #size of the maze
mode = 'lstm' #either 'rnn' or 'lstm' -> you also need to switch it in models.py in the config dictionary
seed = 0 #seed for the training

#Put the seed
np.random.seed(seed)

#Generates the fixed maze with the given seed
generate_and_save_maze((size,size))

from ppoagent import agent,config

import time
#Things to print while training
l = ['episode_len_mean','episode_reward_max','episode_reward_mean','episode_reward_min','time_this_iter_s']
#Prepare statistics lists
reward_mean = []
len_mean = []
#Train
for i in range(1500):
    result = agent.train()
    print(i)
    for k in l:
        print(k,result[k])
    reward_mean.append(result['episode_reward_mean'])
    len_mean.append(result['episode_len_mean'])
    #print(pretty_print(result))

#Saves checkpoint
name = str(size)+'-'+mode+'-'+'checkpoint-'+str(time.time()).split('.')[0]
agent.save(name)

print('SAVED',name)

#Save things
import os
import matplotlib.pyplot as plt

directory = os.path.join('imgs',name)
os.makedirs(directory,exist_ok=True)

#Save config file
import json

config_file = open(os.path.join(directory,'config.json'),'w')
#json.dump(config,config_file)

#Save perf figure
fig, ax1 = plt.subplots()

ax1.set_ylabel('mean_reward', color='orange')
ax1.plot(range(len(reward_mean)),reward_mean,label='mean reward',color='orange')
ax1.tick_params(axis='y', labelcolor='orange')

ax2 = ax1.twinx()

ax2.set_ylabel('mean_length', color='blue')
ax2.plot(range(len(len_mean)),len_mean,label='mean length',color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

fig.tight_layout()

fig_name = os.path.join(directory,'training')
plt.savefig(fig_name+'.png')

plt.xlabel('nb_episodes')
plt.legend()
plt.show()