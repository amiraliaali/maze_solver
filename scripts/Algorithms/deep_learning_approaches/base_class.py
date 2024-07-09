import random
import torch
from maze import Maze, actions_mapping, override
from torch import nn as nn
import numpy as np
import copy

class ReplayMemory:
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def insert(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        assert self.can_sample(batch_size)

        batch = random.sample(self.memory, batch_size)

        # [[s, a, r, s'], [s, a, r, s'], [s, a, r, s']] -> [[s, s, s], [a, a, a], [r, r, r], [s', s', s']]
        batch = zip(*batch) # takes the first element of each list and group them together and so on.
        
        return [torch.cat(items) for items in batch]

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10 # the number of the transitions in the memory should be at least 10 times the batch size

    def __len__(self):
        return len(self.memory)
    


class DeepMazeBaseClass(Maze):
    def __init__(self, original_maze, state_dim, frame_width=500, frame_height=500) -> None:
        super().__init__(frame_width, frame_height)
        self.env = original_maze
        self.state_dim = state_dim
        self.num_actions = len(actions_mapping)
        self.q_network = self.generate_network()
        self.target_q_network = copy.deepcopy(self.q_network).eval()

    def reset(self):
        obs = np.array(self.env.env_reset())
        return torch.from_numpy(obs).unsqueeze(dim=0).float() # adds an extra dimension in the position 0

    def step(self, state, action):
        state = state.numpy().flatten()
        state = tuple(int(x) for x in state)
        action = action.item()
        next_state, reward, done = self.env.next_step(state, action)
        next_state = np.array(next_state)
        next_state = torch.from_numpy(next_state).unsqueeze(dim=0).float()
        reward = torch.tensor(reward).view(1, -1).float() # 0. -> [0.] -> [[0.]]
        done = torch.tensor(done).view(1, -1)
        return next_state, reward, done
    
    def generate_network(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions))

    @override
    def test_agent(self, state):
        next_state = state
        end = False
        while not end:
            frame_copy = np.copy(
                self.empty_maze_frame
            )  # Create a new copy for each iteration

            state_tensor = np.array(next_state)
            state_tensor = torch.from_numpy(state_tensor).unsqueeze(dim=0).float()
            action = torch.argmax(self.q_network(state_tensor)).item()

            next_state, reward, end = self.next_step(next_state, action)
            frame = self.draw_agent(frame_copy, self.cell_size, next_state)
            self.render(frame, end)  # Render the frame_copy
            self.all_frames.append(frame)
    
    @override
    def policy(self, state, epsilon=0.):
        if torch.rand(1) < epsilon:
            return torch.randint(self.num_actions, (1, 1))
        else:
            av = self.q_network(state).detach()
            return torch.argmax(av, dim=-1, keepdim=True)