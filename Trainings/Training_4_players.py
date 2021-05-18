# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 13:49:29 2021

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
file_name="ENV_4_players_RGB", seed=randomseed, side_channels=[channel])
channel.set_configuration_parameters(time_scale=6)  # jouer avec la valeur
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
TRAINING---------------------------------------------------------------------------------------------------------------
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
        initial_channels = 3  # 3 pour RGB, 1 pour grayscale
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


BUFFER_SIZE = 15000     # replay buffer size    
BATCH_SIZE = 64         # minibatch size 
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 0.00025            # learning rate
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
            return np.random.randint(0, 4, size=4)
            
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


        actions = actions.reshape(256, 1)
        
        #print("self qnetwork: ", self.qnetwork_local(states).shape)
        #print('actions: ', actions.shape)     #les actions sont des float        
        
        predicted_targets = self.qnetwork_local(states).gather(1, actions)

        with torch.no_grad():
            labels_next = self.qnetwork_target(
                next_states).detach().max(1)[0].unsqueeze(1)

        # .detach() ->  Returns a new Tensor, detached from the current graph.

    

        #print('labels_next: ', labels_next.shape)
        #print('rewards: ', rewards.shape)
        #print('dones: ', dones.shape)
        

        
        rewards = rewards.reshape(256, 1)
        dones = dones.reshape(256, 1)
        #labels_next = labels_next.reshape(64, 8)
        labels = rewards + (gamma * labels_next*(1-dones))

        #print('rewards : ', rewards)
        #print('dones : ', dones)
        
        #print('predicted targets: ', predicted_targets.shape)

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

#POUR LE CURRICULUM LEARNING: ON ENTRAINE SUR BASE DU MODELE QUI EST CAPABLE DE MANGER LES PELLETS :
#agent.qnetwork_local.load_state_dict(torch.load('C:/Users/Utilisateur/Desktop/MA1/Q2/Projet environnements/ENTRAINEMENTS DE PAQUES/Pellet eating grayscale/Training_Pellet_Eating_grayscale_final_episode_reset.pth'))

def dqn(env, n_episodes=2500, max_t=1000, eps_start = 1, eps_end=0.01, eps_decay=0.999):
    
    scores = []
    agents_scores = []
    scores_window = deque(maxlen=100)
 
    eps = eps_start

    MaxScore = 0
    
    # WA1 = []
    # WA2 = []
    # WB1 = []
    # WB2 = []
    
    # LA1 = []
    # LA2 = []
    # LB1 = []
    # LB2 = []

    
    for i_episode in range(1, n_episodes+1):
        
        # LosersVector = [0,0,0,0]
        # WinnersVector = [0,0,0,0]

        env.reset()  
        #tmp = [0, 0, 0, 0]
        

        # on prend les infos de l'env (state)
        behavior_name = list(env.behavior_specs)[0]
        spec = env.behavior_specs[behavior_name]
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        # l'input du réseau de neurones
        state = decision_steps.obs[0]

        masses = decision_steps.obs[1].reshape(4)

        #print('masses shape: ', masses.shape) #(8,)
        #print('masse: ', masses)
        #done = False
        total = np.zeros(len(decision_steps))
               

        for t in range(max_t):

            #print('masses: ', masses)
            # on choisit l'action
            action = agent.act(state, eps)
            
            
            action_full = np.zeros((4, 4), dtype = int)
            
            for i in range(len(action_full)):   #2
                action_full[i][0] = action[i]

            """
            action_rnd = np.random.randint(0, 4, size = 4).reshape(4, 1)
            
            for cnt in range(2,4):
                action_full[cnt][0] = action_rnd[cnt]
            """
            action_tuple = ActionTuple()
            action_tuple.add_continuous(action_full)

            # on effectue l'action et on récupère next_state, reward et done
            env.set_actions(behavior_name, action_tuple)
            env.step()
            next_behavior_name = list(env.behavior_specs)[0]
            next_spec = env.behavior_specs[next_behavior_name]
            next_decision_steps, next_terminal_steps = env.get_steps(
                next_behavior_name)
            next_state = next_decision_steps.obs[0]
            next_masses = next_decision_steps.obs[1].reshape(4)
            
            
            #Si l'agent a gagné en masse, on considère que done = True 

            
            #print('\n masses: ', next_masses)
    
            reward = [0, 0, 0, 0]
            done = [False, False, False, False]
            
            
            
            # 1er Système de reward:
            #   S'il mange un pellet : +0.5
            #   S'il mange son adversaire : +1
            #   S'il ne mange rien : -0.01
            #   S'il se fait manger : -1
            
            for k in range(len(masses)):                
                  
                if (next_masses[k] > masses[k]):
                    #si sa masse a augmenté:
                    done[k] = True
                        
                    if (next_masses[k] - masses[k] >= 10):
                        if(masses[k] == 0):
                            #cas où il spawn simplement
                            reward[k] = -0.01
                            done[k] = False
   
                        else:
                            #si l'agent a mangé l'autre:
                            reward[k] = 1
                            #WinnersVector[k] += 1

                    else:
                        #s'il a mangé un pellet
                        reward[k] = 0.4 #0.4
                    
                elif(next_masses[k] == 0 and masses[k] !=0):
                    #Si l'agent s'est fait manger
                    reward[k] = -0.5 #-0.5
                    #LosersVector[k] += 1
                    
                    
                else:
                    #si l'agent ne mange rien et ne se fait pas manger
                    reward[k] = -0.01
                
            
            
            """
            #METHODE DE REWARD BASEE SUR LA MASSE !!!!!
            for k in range(len(masses)): 

                if(next_masses[k] >= 10 and masses[k] == 0):
                    #cas du respawn
                    reward[k] = 0
                else:
                    reward[k] = next_masses[k] - masses[k]
                    
                    if(reward[k] > 0):
                       done[k] = True
            """         

                    
                    

            """
            #   SYSTEME DE REWARD V3:
            #   En théorie ça permettra à l'agent d'accorder de moins en moins d'importance aux pellets 
            #   et de + en + aux petits joueurs à mesure qu'il grossit.
            #   S'il mange un pellet : +0.1
            #   S'il mange son adversaire : +1
            #   S'il ne mange rien : -(0.2/100)*masse
            #   S'il se fait manger : -1
            
            for k in range(len(masses)):                  
                if (next_masses[k] > masses[k]):
                    #si sa masse a augmenté:
                    done[k] = True
                        
                    if (next_masses[k] - masses[k] >= 10):
                        if(next_masses[k] == 10):
                            #cas où il spawn simplement
                            reward[k] = 0
                            done[k] = False
   
                        else:
                            #si l'agent a mangé l'autre:
                            reward[k] = 1

                    
                    else:
                        #s'il a mangé un pellet
                        reward[k] = 0.1
                    
                elif(next_masses[k] == 0 and masses[k] !=0):
                    #Si l'agent s'est fait manger
                    reward[k] = -1
                    
                else:
                    #si l'agent ne mange rien et ne se fait pas manger
                    #0.2/(100*5) car la baisse  de masse a lieu tous les 5 steps donc on étale le penalty sur tous les 
                    #steps pour plus d'homogénéité
                    reward[k] = -0.01
            
            """
            #print('masses: ', masses)
            #print('tmp: ', tmp)
            #print('rewards: ', reward)
            #print('dones: ', done)
            
            #print('losers: ', LosersVector)
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

        if np.mean(total) > MaxScore:
            # SAUVEGARDER LE MODELE QUAND LE SCORE EST MAXIMUM !!
            MaxScore = np.mean(total)
            torch.save(agent.qnetwork_local.state_dict(), 'Training_masses_random_grayscale_curriculum bis.pth')


        # on sauve le score le + récent du 1er agent
        scores_window.append(total)
        scores.append(np.mean(total))
        
        agents_scores.append(total)
        

        eps = max(eps*eps_decay, eps_end)  # on réduit epsilon
        
        """
        WA1.append(WinnersVector[0])
        WA2.append(WinnersVector[1])
        WB1.append(WinnersVector[2])
        WB2.append(WinnersVector[3])
        
        LA1.append(LosersVector[0])
        LA2.append(LosersVector[1])
        LB1.append(LosersVector[2])
        LB2.append(LosersVector[3])
        """
        #print('WB2:', WB2)
        #print('winners: ', WinnersVector)

        print('\rEpisode {}\tAverage Score {:.2f}'.format(
            i_episode, np.mean(scores_window)), end="")

    
    torch.save(agent.qnetwork_local.state_dict(), 'Training_final_masses_random_grayscale_curriculum bis.pth')
    return scores, agents_scores#,  WA1, WA2, WB1, WB2, LA1, LA2, LB1, LB2

scores, agents_scores = dqn(env)  #, WA1, WA2, WB1, WB2, LA1, LA2, LB1, LB2

#print('scores out: ', scores )
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.title('average scores')
plt.ylabel('Score')
plt.xlabel('Epsiode #')
plt.savefig('Average_score_4_players_Training_masses_random_grayscale_curriculum bis.png')
plt.show()



for i in range(4):
    plt.plot([item[i] for item in agents_scores])
    plt.title('scores of agent {}'.format(i))
    plt.ylabel('Score')
    plt.xlabel('Epsiode #')
    #plt.savefig('4 players Training_masses_random_grayscale_curriculum bis Scores of agent {}.png'.format(i))
    plt.show()



print('Mean score: ', np.mean(scores))


print('Max score: ', max(scores))


max_value, max_index = max((x, (i, j))
                            for i, row in enumerate(agents_scores)
                            for j, x in enumerate(row))


print('Max individual score: ', max_value)


scores_window = deque(maxlen=100)
fenetre = []

for i in range(len(scores)):
    scores_window.append(agents_scores[i])
    fenetre.append(np.mean(scores_window))
    
plt.figure()
plt.plot(scores, c='b', label = 'Scores')
plt.plot(fenetre, c='r', label = 'Avg on 100 episodes')
plt.title('')
plt.xlabel('Episode #')
plt.ylabel('Score')
plt.legend()
plt.show()



#SAVE AND LOAD LISTS
data = np.array(agents_scores)
np.savez("training", data)





# #############################
# #####OBSERVING THE AGENT#####
# #############################



# agent = Agent(state_size = (84, 84), action_size = 4, seed = 0)

# # load the weights from file
# agent.qnetwork_local.load_state_dict(torch.load('C:/Users/Utilisateur/Desktop/MA1/Q2/Projet environnements/ENTRAINEMENTS DE PAQUES/4 PLAYERS ENTRAINEMENT 18 CURRICULUM GRAY/Training_final_masses_rewardMasse_random_grayscale_curriculum.pth'))


# #load the weights from file

# names = ['A1', 'A2', 'bot1', 'bot2']
# LosersVector = [0,0,0,0]
# WinnersVector = [0,0,0,0]

# WA1 = []
# WA2 = []
# WB1 = []
# WB2 = []

# LA1 = []
# LA2 = []
# LB1 = []
# LB2 = []

# for i in range(150):
#     env.reset()
    
#     behavior_name = list(env.behavior_specs)[0] 
#     spec = env.behavior_specs[behavior_name]        
#     decision_steps, terminal_steps = env.get_steps(behavior_name)

    
#     state = decision_steps.obs[0] #j'ai mon state
#     masses = decision_steps.obs[1].reshape(4)
    
#     #print('episode ', i, WinnersVector)
#     print('\rEpisode {}\tLosers {}'.format(i, LosersVector), end="")
    


#     #print('winning: ', WA2)

#     for j in range(1000):
        
#         action = agent.act(state)
#         action_full = np.zeros((4, 4), dtype = int)
        
#         for m in range(2): #2
#             action_full[m][0] = action[m]
        
#         #print('action_full avant rnd: ', action_full)
        
#         action_0 = np.random.randint(0, 4, size = 4).reshape(4, 1)
        
#         for n in range(2,4):
#             action_full[n][0] = action_0[n]
        
#         #print('action_full après rnd: ', action_full)
        
        
#         action_tuple = ActionTuple()
#         action_tuple.add_continuous(action_full)
        
#         """
#         plt.imshow(state.squeeze()[0]) #ils ont l'air de tous bouger
#         #plt.title("state")
#         plt.show()
#         """
#         env.set_actions(behavior_name, action_tuple)
#         env.step()
#         next_behavior_name = list(env.behavior_specs)[0] 
#         next_spec = env.behavior_specs[next_behavior_name]
#         next_decision_steps, next_terminal_steps = env.get_steps(next_behavior_name)
#         next_state = next_decision_steps.obs[0]
#         next_masses = next_decision_steps.obs[1].reshape(4)
        

#         for k in range(len(masses)):
#             if(next_masses[k] == 0 and masses[k] !=0):
#                 LosersVector[k] += 1
                
#             elif(next_masses[k] - masses[k] >= 10 and masses[k] != 0):
#                 WinnersVector[k] +=1

#         #on met à jour le state
#         state = next_state    
#         spec = next_spec
#         decision_steps = next_decision_steps
#         terminal_steps = next_terminal_steps
#         masses = next_masses
        
        
        
#     WA1.append(WinnersVector[0])
#     WA2.append(WinnersVector[1])
#     WB1.append(WinnersVector[2])
#     WB2.append(WinnersVector[3])
    
#     LA1.append(LosersVector[0])
#     LA2.append(LosersVector[1])
#     LB1.append(LosersVector[2])
#     LB2.append(LosersVector[3])


# plt.plot(names, LosersVector)
# plt.title('Number of times agents were eaten')
# plt.show()


# plt.bar(names, WinnersVector)
# plt.title('Number of times agent ate an opponent')
# plt.show()


env.close()
