# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:36:31 2021

@author: Lucien

https://unnatsingh.medium.com/deep-q-network-with-pytorch-d1ca6f40bfda
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel


from math import floor
from typing import Tuple
from skimage.color import rgb2gray
from collections import namedtuple, deque


"""
# Initalise the launcher , get variable names , shapes etc.
engine_configuration_channel = EngineConfigurationChannel()
env = UnityEnvironment(file_name="AGARIO_ENV2", side_channels = [engine_configuration_channel], base_port=3053)

"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")



randomseed = random.randint(0, 20000)



#env = UnityEnvironment(file_name="C:/Users/Utilisateur/Desktop/MA1/Q2/BIS-ML-AgarISIA-development/ENV_AGARIO", seed=1, side_channels=[])
channel = EngineConfigurationChannel()
env = UnityEnvironment(
    file_name="ENV_Solo_Pellet_Eating_RGB", seed=randomseed, side_channels=[channel])
channel.set_configuration_parameters(time_scale=8)  # jouer avec la valeur
env.reset()

"""
NbrBehaviors = len(list(env.behavior_specs))
print(f"Number of behaviors : {NbrBehaviors}")

behavior_name = list(env.behavior_specs)[0]
print(f"Name of the behavior : {behavior_name}")
spec = env.behavior_specs[behavior_name]
"""

"""
EXAMINING THE STATE AND ACTION SPACE
"""


print('seed: ', randomseed)

"""
# Examine the number of observations per Agent
print("Number of observations : ", len(spec.observation_shapes))

print('Number of agents: ', len(env.behavior_specs))

# Is there a visual observation ?
# Visual observation have 3 dimensions: Height, Width and number of channels
vis_obs = any(len(shape) == 3 for shape in spec.observation_shapes)
print("Is there a visual observation ?", vis_obs)


# Is the Action continuous or multi-discrete ?
if spec.action_spec.continuous_size > 0:
    print(f"There are {spec.action_spec.continuous_size} continuous actions")
    action_size = spec.action_spec.continuous_size

if spec.action_spec.is_discrete():
    print(f"There are {spec.action_spec.discrete_size} discrete actions")
    action_size = spec.action_spec.discrete_size

# How many actions are possible ?
#print(f"There are {spec.action_size} action(s)")

# For discrete actions only : How many different options does each action has ?
if spec.action_spec.discrete_size > 0:
    for action, branch_size in enumerate(spec.action_spec.discrete_branches):
        print(f"Action number {action} has {branch_size} different options")


observation_space = spec.observation_shapes[0]
# action_space = spec.action_spec.discrete_branches[0] #shape[0] discrete_action_branches

# Get the state of the agents
step_result = env.get_steps(behavior_name)

# get the steps from the environment
decision_steps, terminal_steps = env.get_steps(behavior_name)

state = (decision_steps.obs[0])  # state shape:  (8, 84, 84, 3) RGB
print("state shape: ", state.shape)


#print("masses: ", decision_steps.obs[1].shape)

for i in range(len(state)):
    plt.imshow(state[i].squeeze())
    plt.title('un state')
    plt.show()
"""

"""
DEPLACEMENTS ALEATOIRES: OK CA MARCHE
"""

"""
while True:
    
    behavior_name = list(env.behavior_specs)[0] 
    spec = env.behavior_specs[behavior_name]        
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    
    tracked_agent = -1 # -1 indicates not yet tracking
    state = (decision_steps.obs[0]).squeeze()
    
    


    plt.imshow(state)
    plt.title('6e state')
    plt.show()
    
    #print('state shape: ', state.shape)

    action_0 = np.random.randint(0, 4, size = 1)#.reshape(8, 1)

    print('action random: ', action_0)
    
    #action = np.array([action_0, 0, 0, 0]).reshape(1, 4)
    
    action = np.zeros((4), dtype = int)

    action[0] = action_0
    
    
    action_tuple = ActionTuple()
    action_tuple.add_continuous(action.reshape(1,4))

    #on effectue l'action et on récupère next_state, reward et done
    env.set_actions(behavior_name, action_tuple)
    env.step()
    
    next_behavior_name = list(env.behavior_specs)[0] 
    next_spec = env.behavior_specs[next_behavior_name]
    next_decision_steps, next_terminal_steps = env.get_steps(next_behavior_name)
    next_state = (next_decision_steps.obs[0]).squeeze()


    reward = []

    
    
    for TAgents in range (len(decision_steps)):      
        tracked_agent = decision_steps.agent_id[TAgents]

        if tracked_agent  in next_decision_steps: # The agent requested a decision
          reward_temp = next_decision_steps[tracked_agent].reward
                    
        if tracked_agent in next_terminal_steps: # The agent terminated its episode
          reward_temp = next_terminal_steps[tracked_agent].reward      

    #print('reward: ', reward_temp)
    
    
    #on met à jour le state
    state = next_state    
    spec = next_spec
    decision_steps = next_decision_steps
    terminal_steps = next_terminal_steps
    
"""
    
    
"""
TRAINING
"""





class DQN_bis(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], encoding_size: int, seed: int):
        """
        input_shape : (84, 84, 3) = dimensions de l'image en couleurs
        encoding_size : 126
        output_size : 4 (action_size) (haut bas gauche droite)
        """
        super(DQN_bis, self).__init__()
        height = input_shape[0]
        width = input_shape[1]

        #initial_channels = input_shape[2]
        initial_channels = 3  # 3 for RGB, 1 for grayscale
        output_size = 4  # 4 actions

        conv_1_hw = self.conv_output_shape((height, width), 8, 4) # = (20, 20)
        conv_2_hw = self.conv_output_shape(conv_1_hw, 4, 2)       # = (9, 9)

        
        self.seed = torch.manual_seed(seed)

        self.final_flat = conv_2_hw[0] * conv_2_hw[1] * 32   #9*9*32 = 2592
        

        

        self.conv1 = nn.Conv2d(initial_channels, 16, [8, 8], [4, 4])
        self.conv2 = nn.Conv2d(16, 32, [4, 4], [2, 2])

        self.dense1 = nn.Linear(self.final_flat, encoding_size)  #on a en entrée 2592 et en sortie 126
        self.dense2 = nn.Linear(encoding_size, output_size)      #on a en entrée 126 et en sortie 4

    def forward(self, visual_obs: torch.tensor):
        visual_obs = visual_obs.permute(0, 3, 1, 2)
        #print('visual obs: ', visual_obs.shape)
        conv_1 = F.relu(self.conv1(visual_obs))
        conv_2 = F.relu(self.conv2(conv_1))
        hidden = self.dense1(conv_2.reshape([-1, self.final_flat]))
        hidden = F.relu(hidden)
        hidden = self.dense2(hidden)
        # print('hidden: ', hidden.shape)  [9, 5] ok
        return hidden

    @staticmethod
    def conv_output_shape(h_w: Tuple[int, int], kernel_size: int = 1, stride: int = 1, pad: int = 0, dilation: int = 1,):
        """
        Computes the height and width of the output of a convolution layer.
        """
        h = floor(
            ((h_w[0] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)

        w = floor(
            ((h_w[1] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)

        return h, w


BUFFER_SIZE = 30000  # replay buffer size    peut être descendre à 10 000 car 100 000 c'est trop pour mon processeur
BATCH_SIZE = 32     # minibatch size
GAMMA = 0.99             # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 0.00025               # learning rate je l'ai desendue un peu mettre à 0.00025 pour voir ?
UPDATE_EVERY = 4        # how often to update the network


class Agent():
    """ Interacts with and learns from the environment"""

    def __init__(self, state_size, action_size, seed):
        """
        Initialize an agent object

        Parameters
        ----------
        state_size : 
            Dimension of each state. (ici: [84, 84])
        action_size : 
            dimension of each action.
        seed : int
            Random seed.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-network
        self.qnetwork_local = DQN_bis(state_size, 256, seed).to(device)
        self.qnetwork_target = DQN_bis(state_size, 256, seed).to(device)
        
        #self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        #self.optimizer = optim.SGD(self.qnetwork_local.parameters(), lr=LR, momentum = 0.9)
        self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=LR, alpha=0.95, eps=0.01, weight_decay=0, momentum=0.95, centered=False)


        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_step, done):

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_step, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step+1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn

            if len(self.memory) > BATCH_SIZE:
                experience = self.memory.sample()
                self.learn(experience, GAMMA)

    def act(self, state, eps=0):
        """Returns action for given state as per current policy
        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        # On récupère les Q-values
        state = torch.from_numpy(state).float().to(device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            #print('Q-values: ', action_values)
            
            #action_values = (self.qnetwork_local(torch.from_numpy(state)).detach().numpy())
        self.qnetwork_local.train()

        #print('action shape act: ', action_values.shape)
        # Epsilon-greedy action selection
        if random.random() > eps:
            # return np.argmax(action_values.cpu().data.numpy())
            #actions = np.argmax(action_values.cpu().data.numpy())

            action = np.argmax(action_values.cpu(), axis=1).data.numpy()
            #print('action values: ', action_values)
            #print('chosen action: ', action)   #ok
            #print('action Q')
            return action

        else:
            # print('action non random') #ça a l'air de bien passer
            # return spec.action_spec.random_action(len(decision_steps)) #action random
            #print('action random')
            return np.random.randint(0, 4, size=1)
            
            # return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # TODO: compute and minimize the loss
        criterion = torch.nn.MSELoss()
        self.qnetwork_local.train()
        self.qnetwork_target.eval()
        # shape of output from the model (batch_size,action_dim) = (64,4)
        # chez moi c'est ((84, 84), 4)
        #print('actions shape: ', actions.shape)
        #print('qnet shape: ', self.qnetwork_local(states).shape)

        #actions = actions.reshape(512, 1)

        #print("self qnetwork: ", self.qnetwork_local(states).shape)
        # print('actions: ', actions.shape)     #les actions sont des float

        predicted_targets = self.qnetwork_local(states).gather(1, actions)

        with torch.no_grad():
            labels_next = self.qnetwork_target(
                next_states).detach().max(1)[0].unsqueeze(1)

        # .detach() ->  Returns a new Tensor, detached from the current graph.

        # print('labels_next: ', labels_next.shape)
        # print('rewards: ', rewards.shape)
        # print('dones: ', dones.shape)

        #labels_next = labels_next.reshape(64, 8)
        labels = rewards + (gamma * labels_next*(1-dones))

        #predicted_targets = predicted_targets.reshape(64, 8)
        loss = criterion(predicted_targets, labels).to(device)

        #print('loss: ', loss)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        
        
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1-tau)*target_param.data)


class ReplayBuffer:
    """ fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                                 "action",
                                                                 "reward",
                                                                 "next_state",
                                                                 "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


agent = Agent(state_size=(84, 84), action_size=4, seed=0)

#2500 épisodes de 1000 steps ça va faire long
def dqn(env, n_episodes=2500, max_t=100, eps_start = 1, eps_end=0.01, eps_decay=0.999):
    
    scores = []
    #agents_scores = []
    scores_window = deque(maxlen=100)
 
    eps = eps_start

    MaxScore = 0

    for i_episode in range(1, n_episodes+1):

        env.reset()  # ça a pas l'air de bien reset -> normal 

        # on prend les infos de l'env (state)
        behavior_name = list(env.behavior_specs)[0]
        spec = env.behavior_specs[behavior_name]
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        #tracked_agent = -1  # -1 indicates not yet tracking
        # l'input du réseau de neurones, 8 images en gris (8, 84, 84, 1)
        state = decision_steps.obs[0]

        masses = decision_steps.obs[1]

        #print('masses shape: ', masses.shape) #(8,)
        #print('masse: ', masses)
        #done = False
        total = 0
        
        """
        plt.imshow(state.squeeze())
        plt.title('state')
        plt.show()
        """
        
        for t in range(max_t):

            if(t>20 and t<40):
                plt.imshow(state.squeeze()[0]) #ils ont l'air de tous bouger
                plt.title("state")
                plt.show()
            
            #print('masses: ', masses)
            # on choisit l'action
            action = agent.act(state, eps)
            
            
            action_full = np.zeros((4), dtype = int)
            action_full[0] = action


            action_tuple = ActionTuple()
            action_tuple.add_continuous(action_full.reshape(1,4))

            # on effectue l'action et on récupère next_state, reward et done
            env.set_actions(behavior_name, action_tuple)
            env.step()
            next_behavior_name = list(env.behavior_specs)[0]
            next_spec = env.behavior_specs[next_behavior_name]
            next_decision_steps, next_terminal_steps = env.get_steps(
                next_behavior_name)
            next_state = next_decision_steps.obs[0]

            next_masses = next_decision_steps.obs[1]
            
            
            #Si l'agent a mangé le pellet (seule manièe de gagner en masse), on considère que done = True 
            
            """
            plt.imshow(next_state.squeeze())
            plt.title('state')
            plt.show()
            """
            #print('\n masses diff: ', next_masses - masses)
    
            
            done = False
            if(next_masses > masses and next_masses - masses < 9):
                    
                #print('if diff: ', next_masses - masses)
                reward = 1
                done = True    
                 
            else:
                reward = -0.01
                
            
            #print('reward: ', reward)
            #print('masse: ', next_masses)
            #print('done: ', done_vect)

            # On choisit si on va train (learn) le réseau de neurones
            # ou si on va remplir le buffer. Si len(buffer) = batch size on va
            # entrainer le réseau sinon on va ajouter un experience tuple dans le buffer
            agent.step(state, action, reward, next_state, done)

            # on met à jour le state
            state = next_state
            spec = next_spec
            decision_steps = next_decision_steps
            terminal_steps = next_terminal_steps
            masses = next_masses

            total += reward
            # cumulative_reward.append(reward)

        if total > MaxScore:
            # SAUVEGARDER LE MODELE QUAND LE SCORE EST MAXIMUM !!
            MaxScore = total
            torch.save(agent.qnetwork_local.state_dict(), 'Training_Pellet_Eating_grayscale_reset.pth')

        #print('total out: ', total)
        #print('mean total: ', np.mean(total))
        # on sauve le score le + récent du 1er agent
        scores_window.append(total)
        scores.append(total)
        
        #print('window ',scores_window)
        
        #print('scores: ', scores)
        
        
        #print('scores window: ', scores_window)

        eps = max(eps*eps_decay, eps_end)  # on réduit epsilon

        print('\rEpisode {}\tAverage Score {:.2f}'.format(
            i_episode, np.mean(scores_window)), end="")

            
        """
        if i_episode %100==0:
            print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)))
            
        if np.mean(scores_window[0])>=20.0:
            print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
        
            torch.save(agent.qnetwork_local.state_dict(),'checkpoint.pth')
            
            break
        """
    
    torch.save(agent.qnetwork_local.state_dict(), 'Training_Pellet_Eating_grayscale_final_episode_reset.pth')
    return scores

scores = dqn(env)

#print('scores out: ', scores )
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.title('average scores')
plt.ylabel('Score')
plt.xlabel('Epsiode #')
plt.savefig('Average_score_Pellet_Eating_grayscale_reset.png')
plt.show()


print('Mean score: ', np.mean(scores))

print('Max score: ', max(scores))


scores_window = deque(maxlen=100)
fenetre = []

for i in range(len(scores)):
    scores_window.append(scores[i])
    fenetre.append(np.mean(scores_window))
    
plt.figure()
plt.plot(scores, c='b', label = 'Scores')
plt.plot(fenetre, c='r', label = 'Avg on 100 episodes')
plt.title('')
plt.xlabel('Episode #')
plt.ylabel('Score')
plt.legend()
plt.savefig('Average_score_Pellet_Eating_RGB.png')
plt.show()




#SAVE AND LOAD LISTS
data = np.array(scores)
np.savez("Scores RGB", data)



# #############################
# #####OBSERVING THE AGENT#####
# #############################



# agent = Agent(state_size = (84, 84), action_size = 4, seed = 0)

# # load the weights from file
# agent.qnetwork_local.load_state_dict(torch.load('C:/Users/Utilisateur/Desktop/MA1/Q2/Projet environnements/ENTRAINEMENTS DE PAQUES/Pellet eating grayscale/Training_Pellet_Eating_grayscale_final_episode_reset.pth'))


# #load the weights from file

# for i in range(3):
#     env.reset()
    
#     behavior_name = list(env.behavior_specs)[0] 
#     spec = env.behavior_specs[behavior_name]        
#     decision_steps, terminal_steps = env.get_steps(behavior_name)

    
#     state = decision_steps.obs[0] #j'ai mon state
    


#     for j in range(1000):
#         action = agent.act(state)
            
            
#         action_full = np.zeros((4), dtype = int)
#         action_full[0] = action


#         action_tuple = ActionTuple()
#         action_tuple.add_continuous(action_full.reshape(1,4))
        
#         """
#         plt.imshow(state.squeeze()) #ils ont l'air de tous bouger
#         plt.title("state")
#         plt.show()
#         """
#         env.set_actions(behavior_name, action_tuple)
#         env.step()
#         next_behavior_name = list(env.behavior_specs)[0] 
#         next_spec = env.behavior_specs[next_behavior_name]
#         next_decision_steps, next_terminal_steps = env.get_steps(next_behavior_name)
#         next_state = next_decision_steps.obs[0]
        
#         #on met à jour le state
#         state = next_state    
#         spec = next_spec
#         decision_steps = next_decision_steps
#         terminal_steps = next_terminal_steps






env.close()
