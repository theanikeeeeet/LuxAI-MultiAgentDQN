import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os 
class SharedReplayBuffer:
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.state_memory = np.zeros((max_size, *input_dims), dtype=np.float32)
        self.next_state_memory = np.zeros((max_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(max_size, dtype=np.int32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.terminal_memory = np.zeros(max_size, dtype=np.bool)

    def store_transition(self, state, next_state, action, reward, done):
        index = self.mem_ctr % self.mem_size
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_ctr += 1

    def sample_batch(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        return (
            self.state_memory[batch],
            self.next_state_memory[batch],
            self.action_memory[batch],
            self.reward_memory[batch],
            self.terminal_memory[batch],
        )



class DeepQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, lr=0.001):
        super(DeepQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 64)

        # Q-values for 5 discrete actions (0-4 move, 5=sap)
        self.q_values = nn.Linear(64, n_actions)

        # Output for (x, y) coordinates (only for sap action)
        self.x_coordinate = nn.Linear(64, 4)  # Predict X coordinate (Assuming a 4x4 grid)
        self.y_coordinate = nn.Linear(64, 4)  # Predict Y coordinate (Assuming a 4x4 grid)

        self.softmax = nn.Softmax(dim=-1)  # Softmax for coordinate selection
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        
        q_values = self.q_values(x)  # Get Q-values for all actions
        x_coordinates = self.softmax(self.x_coordinate(x))  # Normalize X output
        y_coordinates = self.softmax(self.y_coordinate(x))  # Normalize Y output

        return q_values, x_coordinates, y_coordinates


class MultiAgentDQN:
    def __init__(self, n_agents, input_dims, n_actions, player, opponent_player, 
                 lr=0.001, gamma=0.99, eps=1.0, eps_min=0.01, eps_dec=1e-4, 
                 mem_size=10000, batch_size=64, tau=0.005, target_update_freq=5):
        self.n_agents = n_agents
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.tau = tau  # Soft update factor for target network
        self.target_update_freq = target_update_freq  # Update target every few episodes
        self.update_counter = 0  # Track steps for target update

        self.player = player
        self.opp_player = opponent_player
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        # **Create Policy and Target Networks**
        self.policy_agents = [DeepQNetwork(input_dims, n_actions, lr).to(self.device) for _ in range(n_agents)]
        self.target_agents = [DeepQNetwork(input_dims, n_actions, lr).to(self.device) for _ in range(n_agents)]

        # **Load saved models if available**
        for idx, agent in enumerate(self.policy_agents):
            model_path = f'/DATA/Shashank/LuxAis3/Agents/agent_{self.player}/agent_{idx}.pth'
            if os.path.exists(model_path):
                agent.load_state_dict(torch.load(model_path, map_location=self.device))
            agent.to(self.device)

        # **Sync target networks with policy networks initially**
        for target_agent, policy_agent in zip(self.target_agents, self.policy_agents):
            target_agent.load_state_dict(policy_agent.state_dict())
            target_agent.to(self.device)

        # **Replay Buffer**
        self.replay_buffer = SharedReplayBuffer(mem_size, input_dims)

    def _update_target_network(self):
        """Soft update target network using Polyak averaging and save policy network weights"""
        for idx, policy_net in enumerate(self.policy_agents):
            torch.save(policy_net.state_dict(), f'/DATA/Shashank/LuxAis3/Agents/agent_{self.player}/agent_{idx}.pth')
        for target_agent, policy_agent in zip(self.target_agents, self.policy_agents):
            for target_param, policy_param in zip(target_agent.parameters(), policy_agent.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

    def choose_actions(self, states):
        actions, x_coords, y_coords = [], [], []

        for i in range(self.n_agents):
            state = torch.tensor(states[i], dtype=torch.float32).to(self.device)
            q_values, x_probs, y_probs = self.policy_agents[i](state)  # Use policy network
            
            action = torch.argmax(q_values).item()  # Select highest Q-value action

            if action == 5:  # If sap action, sample (x, y)
                x_choice = torch.multinomial(x_probs, 1).item()
                y_choice = torch.multinomial(y_probs, 1).item()
            else:
                x_choice, y_choice = 0, 0  # Default (x, y) for non-sap actions

            actions.append(action)
            x_coords.append(x_choice)
            y_coords.append(y_choice)

        return actions, x_coords, y_coords

    def store_experience(self, states, next_states, actions, rewards, dones, num_available):
        """Stores experience and trains periodically"""
        for i in range(num_available):
            self.replay_buffer.store_transition(states[i], next_states[i], actions[i], rewards[i], dones[i])

        self.train_agents()

    def train_agents(self):
        """Trains the policy network using sampled experiences"""
        if self.replay_buffer.mem_ctr < self.batch_size:
            return

        for policy_agent, target_agent in zip(self.policy_agents, self.target_agents):
            state_batch, next_state_batch, action_batch, reward_batch, done_batch = self.replay_buffer.sample_batch(self.batch_size)

            state_batch_t = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
            next_state_batch_t = torch.tensor(next_state_batch, dtype=torch.float32).to(self.device)
            reward_batch_t = torch.tensor(reward_batch, dtype=torch.float32).to(self.device)
            done_batch_t = torch.tensor(done_batch, dtype=torch.bool).to(self.device)
            action_batch_t = torch.tensor(action_batch, dtype=torch.int64).to(self.device)

            q_values, x_probs, y_probs = policy_agent(state_batch_t)
            q_values_next, _, _ = target_agent(next_state_batch_t)  # Use target network

            q_val_curr = q_values.gather(1, action_batch_t.unsqueeze(1)).squeeze(1)
            q_val_next = q_values_next.max(1)[0].detach()
            q_val_next[done_batch_t] = 0.0
            q_target = reward_batch_t + self.gamma * q_val_next
            q_loss = nn.MSELoss()(q_target, q_val_curr)

            sap_mask = (action_batch_t == 5).float().unsqueeze(1)
            x_target = torch.nn.functional.one_hot(torch.randint(0, 4, (self.batch_size,)), num_classes=4).float().to(self.device)
            y_target = torch.nn.functional.one_hot(torch.randint(0, 4, (self.batch_size,)), num_classes=4).float().to(self.device)

            x_loss = (sap_mask * -torch.sum(x_target * torch.log(x_probs + 1e-9), dim=1)).mean()
            y_loss = (sap_mask * -torch.sum(y_target * torch.log(y_probs + 1e-9), dim=1)).mean()

            total_loss = q_loss + x_loss + y_loss

            policy_agent.optimizer.zero_grad()
            total_loss.backward()
            policy_agent.optimizer.step()

        self.eps = max(self.eps - self.eps_dec, self.eps_min)
        self.update_counter += 1

        # Update the target network periodically
        if self.update_counter % self.target_update_freq == 0:
            self._update_target_network()
