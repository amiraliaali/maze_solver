from maze import Maze, actions_mapping, override
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn as nn
from torch.optim import AdamW
import numpy as np
from maps import maze_map_2
import copy
import random
from tqdm import tqdm

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

class DeepSARSA(Maze):
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
    
    def train_deep_sarsa(self, episodes, alpha=0.001, batch_size=32, gamma=0.99, epsilon=0.05):
        optim = AdamW(self.q_network.parameters(), lr=alpha)
        memory = ReplayMemory()
        stats = {"MSE Loss": [], "Returns": []}

        for episode in tqdm(range(1, episodes+1)):
            state = self.reset()
            done = False
            ep_return = 0

            while not done:
                action = self.policy(state, epsilon)
                next_state, reward, done = self.step(state, action)
                memory.insert([state, action, reward, done, next_state])

                if memory.can_sample(batch_size):
                    state_b, action_b, reward_b, done_b, next_state_b = memory.sample(batch_size)

                    qsa_b = self.q_network(state_b).gather(1, action_b)

                    next_action_b = self.policy(next_state_b, epsilon)
                    next_qsa_b = self.target_q_network(next_state_b).gather(1, next_action_b)

                    target_b = reward_b + ~done_b * gamma * next_qsa_b

                    loss = F.mse_loss(qsa_b, target_b)

                    self.q_network.zero_grad()
                    loss.backward()
                    optim.step()

                    stats["MSE Loss"].append(loss.item())
                
                state = next_state
                ep_return += reward.item()

            stats["Returns"].append(ep_return)

            if episode % 10 == 0:
                self.target_q_network.load_state_dict(self.q_network.state_dict())
        return stats

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
        
    @override
    def run_maze(self, maze_map, draw_the_path, output_filename):
        self.set_up_maze(maze_map, draw_the_path)
        self.test_agent((0,0))
        self.create_video_from_frames(self.all_frames, output_filename)


if __name__ == "__main__":
    maze = Maze()
    maze.set_up_maze(maze_map_2, False)
    maze = DeepSARSA(maze, 2)
    maze.train_deep_sarsa(episodes=500)
    maze.run_maze(maze_map_2, False, "output_video.mp4")
