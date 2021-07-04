import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
END_TRAIN_THRESHOLD = 16.0 


EPS = 1e-8              # PER epsilon
ALPHA = 0.6             # PER alpha
BETA_INIT = 0.4         # PER beta
BETA_INCREASE = 0.005   # PER beta_increase
BETA_MAX = 1.0          # PER beta_max

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):

        ################## PER ##################
        # calculate TD error 
        state_tensor = torch.tensor(state).unsqueeze(dim=0).float().to(device)
        action_tensor = torch.tensor([action]).unsqueeze(dim=0).to(device)
        reward_tensor = torch.tensor(reward).unsqueeze(dim=0).float().to(device)
        next_state_tensor = torch.tensor(next_state).unsqueeze(dim=0).float().to(device)
        done_tensor = torch.tensor(done).unsqueeze(dim=0).float().to(device)
        
        best_action = torch.argmax(self.qnetwork_local(next_state_tensor), dim=-1).unsqueeze(dim=-1)
        Q_target_next = torch.gather(input=self.qnetwork_target(next_state_tensor).detach(), dim=-1, index=best_action) # torch.gather : Gathers values along an axis specified by dim.
        # print(self.qnetwork_local(state_tensor).shape)
        # print(action_tensor.shape)
        target = reward_tensor + GAMMA*Q_target_next*(1-done_tensor) ########## if done True then Q_target_next should be zero
        current_value = torch.gather(input=self.qnetwork_local(state_tensor), dim=1, index=action_tensor) # torch.gather : Gathers values along an axis specified by dim.
        TD_error = abs(target - current_value)
        unnormalized_TD_error = pow(TD_error + EPS, ALPHA)

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, unnormalized_TD_error.detach().cpu())
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.PER_sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
        ################## Double DQN ##################
        best_action = torch.argmax(self.qnetwork_local(next_states), dim=-1).unsqueeze(dim=-1)
        Q_target_next = torch.gather(input=self.qnetwork_target(next_states).detach(), dim=-1, index=best_action) # torch.gather : Gathers values along an axis specified by dim.
        ################## Double DQN ##################

        target = rewards + gamma*Q_target_next*(1-dones) ########## if done True then Q_target_next should be zero
        current_value = torch.gather(input=self.qnetwork_local(states), dim=1, index=actions) # torch.gather : Gathers values along an axis specified by dim.

        self.qnetwork_local.zero_grad()
        loss = torch.mean(torch.sqrt(weights * (target - current_value)**2 + 1e-10))
        # loss = F.mse_loss(current_value, target)
        loss.backward()
        self.optimizer.step()    

        ################## PER ##################
        # updatae TD_error_array
        unnormlized_TD_error = pow(abs(target - current_value).detach() + EPS, ALPHA)
        self.memory.update_TD_error_array(unnormlized_TD_error)
        ################## PER ##################

        "*** YOUR CODE HERE ***"  
        

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    ################## PER ##################
    def update_beta(self):
        self.memory.beta = min(self.memory.beta + self.memory.beta_increase, BETA_MAX)
    ################## PER ##################
    
    def train(self, env, n_episodes=1800, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        """
        Double Deep Q-Learning with PER.    
        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """

        # get the default brain
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]

        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        for i_episode in range(1, n_episodes+1):
            env_info = env.reset(train_mode=True)[brain_name] # reset the environment
            state = env_info.vector_observations[0] # get the current state
            score = 0
            for t in range(max_t):
                action = self.act(state, eps)
                env_info = env.step(action)[brain_name]                  # send the action to the environment
                next_state, reward, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    self.update_beta()       ############################# PER beta update #############################
                    break 
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window) >= END_TRAIN_THRESHOLD:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                torch.save(self.qnetwork_local.state_dict(), 'saved_DDQN_PER_agent.pth')
                break
        return scores


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, beta_init=BETA_INIT, beta_increase=BETA_INCREASE):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)


        ################## PER ##################
        self.unnormalized_TD_error_array = np.array([])
        self.beta = beta_init
        self.beta_increase = beta_increase
        ################## PER ##################
    
    def add(self, state, action, reward, next_state, done, unnormalized_TD_error):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        ################## PER ##################
        self.memory.append(e)
        self.unnormalized_TD_error_array = np.append(self.unnormalized_TD_error_array, unnormalized_TD_error) if len(self.unnormalized_TD_error_array) < self.buffer_size else np.append(self.unnormalized_TD_error_array[1:], unnormalized_TD_error)
        ################## PER ##################

    def PER_sample(self):
        """Prioritily sample a batch of experiences from memory."""
        ################## PER ##################
        # sampling based on priority prob
        idx_array = np.arange(len(self.memory))
        prob = self.unnormalized_TD_error_array / sum(self.unnormalized_TD_error_array)
        self.chosen_idx_array = np.random.choice(idx_array, size=self.batch_size, p=prob)
        experiences = [self.memory[idx] for idx in self.chosen_idx_array]

        # calculate importance sampling weight
        weights_all = pow(len(self.memory) * prob, - self.beta)
        weights_all = weights_all / np.max(weights_all)
        weights = torch.tensor([weights_all[idx] for idx in self.chosen_idx_array]).unsqueeze(dim=-1).to(device)

        ################## PER ##################

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones, weights)

    def update_TD_error_array(self, unormlized_TD_error):
        for numer_idx, idx in enumerate(self.chosen_idx_array):
            self.unnormalized_TD_error_array[idx] = unormlized_TD_error[numer_idx]

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)