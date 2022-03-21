import torch
import numpy as np

import matplotlib.pyplot as plt

from ppoagent import agent #stats for dreamer not supported yet as training it doesn't work

from env import LoadedMazeEnv

import os

from env import generate_and_save_maze

generate_and_save_maze((17,17)) #Generates the environment

#Loads the maze
env = LoadedMazeEnv()
env.reset()

#Parameters for the analysis
network_type = 'rnn'
plot_type = '2d'
name = '17-rnn-checkpoint-1647357856'

#Creates directory to save figures
directory = os.path.join('imgs',name)
os.makedirs(directory,exist_ok=True)

#Plot the maze
plt.figure()
env.render()

fig_name = os.path.join(directory,'maze')
plt.savefig(fig_name+'.png')

#Plot the maze with colors
plt.figure()
env.render()

for j in range(len(env.maze.grid)):
        for k in range(len(env.maze.grid[j])):
            env.maze.grid[j][k].color = np.array([j*255/len(env.maze.grid),k*255/len(env.maze.grid),(np.cos((k+j)/len(env.maze.grid))+1)/2*255])

ratio = 1/10
for i in range(100):
    for j in range(len(env.maze.grid)):
        for k in range(len(env.maze.grid[j])):
            for n in env.maze.grid[j][k].neighbours:
                if n:
                    n.color = n.color*(1-ratio)+env.maze.grid[j][k].color*ratio

for j in range(len(env.maze.grid)):
        for k in range(len(env.maze.grid[j])):
            c = np.int32(env.maze.grid[j][k].color)[None]/255
            plt.scatter([j+0.5],[k+0.5],marker='s',c=c)

fig_name = os.path.join(directory,'maze_color')
plt.savefig(fig_name+'.png')

#Loads the agent
agent.restore(name+'\checkpoint_001000\checkpoint-1000')

model = agent.get_policy().model
model = model.cpu()

STATE = []
COLORS = []
ACTIONS = []
POSITIONS = []
print('computing trajectories')
#Computes trajectories
R = 0
nb = 1000 if plot_type=='2d' else 100
for k in range(nb):
    obs = env.reset()
    seqlen = torch.tensor([1])
    state = model.get_initial_state()
    if network_type == 'lstm':
        state = [state[0][None],state[1][None]]
    else:
        state = [state[0][None]]
    #state = [state[0][None].cuda()]
    STATE.append([])
    COLORS.append([])
    ACTIONS.append([])
    POSITIONS.append([])
    for i in range(50):
        if network_type == 'lstm':
            STATE[k].append([state[0].detach().numpy(),state[1].detach().numpy()])
        else:
            STATE[k].append([state[0].detach().numpy()])
        COLORS[k].append(env.position.color)
        POSITIONS[k].append((env.position.x,env.position.y))
        #STATE[k].append([state[0].cpu().detach().numpy()])
        obs = {'obs':torch.tensor([obs])}
        pol,state = model(obs,state,seqlen)
        pol = torch.exp(pol)
        pol /= pol.sum()
        a = torch.argmax(pol)
        ACTIONS[k].append(a)
        obs,r,d,_ = env.step(a)
        R += r
        #env.render()
        if d:
            if network_type == 'lstm':
                STATE[k].append([state[0].detach().numpy(),state[1].detach().numpy()])
            else:
                STATE[k].append([state[0].detach().numpy()])
            COLORS[k].append(env.position.color)
            ACTIONS[k].append(-1)
            POSITIONS[k].append((env.position.x,env.position.y))
            #STATE[k].append([state[0].cpu().detach().numpy()])
            break
print("mean reward of the loaded agent :",R/nb)

states = [np.array(STATE[i])[...,0,0,:] for i in range(len(STATE))]

if network_type == 'lstm':
    states = [np.transpose(np.concatenate(np.transpose(states[i],(1,2,0)),axis=0),(1,0)) for i in range(len(states))]

states_reshaped = np.concatenate(states,axis=0)

if network_type == 'rnn':
    states_fit_reshaped = states_reshaped[:,0,:]

from sklearn.manifold import TSNE
reshaped_projected_data = TSNE(n_components=2 if plot_type=='2d' else 3, learning_rate='auto',init='random').fit_transform(states_fit_reshaped)

projected_data = []
p = 0
for i in range(len(states)):
    projected_data.append(reshaped_projected_data[p:p+len(states[i])])
    p += len(states[i])

from mpl_toolkits import mplot3d
def plot_data(projected_data,mode): #action/position/time
    data = [projected_data[i][:] for i in range(len(projected_data))]
    reshaped_data = np.concatenate(data,axis=0)
    #fig = plt.figure(figsize=(20,20))
    if plot_type == '3d':
        ax = plt.axes(projection='3d')
    maxlen = max([len(projected_data[i]) for i in range(len(projected_data))])
    for i in range(len(projected_data)):
        alpha = 0.1
        lw = 0.1
        for j in range(len(projected_data[i])):
            if mode == 'time':
                c = np.array([[j/len(projected_data[i]),0.,1-j/len(projected_data[i])]])
            elif mode == "position":
                c = np.int32(COLORS[i][j])/256
            elif mode == "action":
                colors = ['red','blue','green','orange','black']
                c = colors[ACTIONS[i][j]]
            elif mode == 'length':
                c = np.array([[len(projected_data[i])/maxlen,1-len(projected_data[i])/maxlen,0]])
            
            if j != len(data[i])-1:
                if plot_type == '3d':
                    #ax.scatter3D([data[i][j][0]],[data[i][j][1]],[data[i][j][2]],c=c,marker='o',alpha=alpha)
                    ax.plot3D([data[i][j][0],data[i][j+1][0]],[data[i][j][1],data[i][j+1][1]],[data[i][j][2],data[i][j+1][2]],c=c,alpha=alpha,lw=lw,marker='o',markevery=2)
                else:
                    #plt.scatter([data[i][j][0]],[data[i][j][1]],c=c,marker='o',alpha=alpha)
                    plt.plot([data[i][j][0],data[i][j+1][0]],[data[i][j][1],data[i][j+1][1]] ,c=c,alpha=alpha,lw=lw,marker='o',markevery=2)

    plt.xlim(reshaped_data[:,0].min(),reshaped_data[:,0].max())
    plt.ylim(reshaped_data[:,1].min(),reshaped_data[:,1].max())
    
    if plot_type == '2d':
        plt.xlabel('axis 1')
        plt.ylabel('axis 2')

import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.colors import ListedColormap

print('plotting trajectories')

#Time plot
plt.subplots(figsize=(12,10))
plot_data(projected_data,'time')

l = np.linspace(0,1,256)[:,None]
colors = np.concatenate([l,l*0,1-l],axis=1)
cmap = ListedColormap(colors)
norm = mpl.colors.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm)
cbar.ax.set_ylabel('Time in the trajectory')

fig_name = os.path.join(directory,'time')
plt.savefig(fig_name+'.png')

#Position plot
plt.figure(figsize=(10,10))
plot_data(projected_data,'position')
fig_name = os.path.join(directory,'position')
plt.savefig(fig_name+'.png')

#Action plot
plt.figure(figsize=(10,10))
plot_data(projected_data,'action')
patchs = []
for name,color in [('left','red'),('down','blue'),('up','green'),('right','orange'),]:
    patchs.append(mpatches.Patch(color=color, label=name))
plt.legend(handles=patchs)
fig_name = os.path.join(directory,'action')
plt.savefig(fig_name+'.png')

#Length plot
plt.figure(figsize=(12,10))
plot_data(projected_data,'length')

l = np.linspace(0,1,256)[:,None]
colors = np.concatenate([l,1-l,l*0],axis=1)
cmap = ListedColormap(colors)
maxlen = max([len(projected_data[i]) for i in range(len(projected_data))])
norm = mpl.colors.Normalize(vmin=0, vmax=maxlen)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm)
cbar.ax.set_ylabel('Lenght of the full trajectory')

fig_name = os.path.join(directory,'length')
plt.savefig(fig_name+'.png')

print('saved figures')

#Show plots
plt.show()