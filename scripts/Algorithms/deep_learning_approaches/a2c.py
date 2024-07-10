from maze import Maze, actions_mapping, override
import os
from base_class import DeepMazeBaseClass
from maps import maze_map_2
import torch
from torch import nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


class ParallelEnv(Maze):
    def __init__(self, envs_list, frame_width, frame_height) -> None:
        super().__init__(frame_width, frame_height)
        self.envs_list = envs_list
        self.actor = self.actor_network()
        self.critic = self.critic_network()

    def reset(self):
        states = []
        for env in self.envs_list:
            states.append(env.reset())
        stacked_states = torch.stack(states)
        # Reshape to flatten the first two dimensions
        flattened_states = stacked_states.view(-1, stacked_states.size(-1))
        return flattened_states
    
    def step(self, states, actions):
        next_states = []
        rewards = []
        dones = []
        assert len(self.envs_list) == len(states) == len(actions)

        for i in range(len(states)):
            next_state, reward, done = self.envs_list[i].step(states[i], actions[i])
            next_states.append(next_state.view(1, -1))  # Convert next_state to 2D
            rewards.append(reward.view(1, -1))          # Convert reward to 2D
            dones.append(done.view(1, -1))              
            
        stacked_next_states = torch.cat(next_states, dim=0)  # Concatenate along batch dimension
        stacked_rewards = torch.cat(rewards, dim=0)          # Concatenate along batch dimension
        stacked_dones = torch.cat(dones, dim=0)              # Concatenate along batch dimension
        
        return stacked_next_states, stacked_rewards, stacked_dones
    
    def actor_network(self):
        return nn.Sequential(
            nn.Linear(self.envs_list[0].state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.envs_list[0].num_actions),
            nn.Softmax(dim=-1))
    
    def critic_network(self):
        return nn.Sequential(
            nn.Linear(self.envs_list[0].state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softmax(dim=-1))

    def actor_critic(self, episodes, alpha=0.0001, gamma=0.99):
        actor_optim = AdamW(self.actor.parameters(), lr=alpha)
        critic_optim = AdamW(self.critic.parameters(), lr=alpha)
        stats = {"Actor Loss" : [], "Critic Loss": [], "Returns" : []}

        for episode in tqdm(range(1, episodes+1)):
            state = self.reset()
            done_b = torch.zeros((len(self.envs_list), 1), dtype=torch.bool)
            ep_return = torch.zeros((len(self.envs_list), 1))
            I = 1.

            while not done_b.all():
                action = self.actor(state).multinomial(1).detach()
                next_state, reward, done = self.step(state, action)

                value = self.critic(state)
                target = reward + ~done * gamma * self.critic(next_state).detach()
                critic_loss = F.mse_loss(value, target)
                self.critic.zero_grad()
                critic_loss.backward()
                critic_optim.step()

                advantage = (target - value).detach()
                probs = self.actor(state)
                log_probs = torch.log(probs + 1e-6)
                action_log_prob = log_probs.gather(1, action)
                entropy = - torch.sum(probs * log_probs, dim=-1, keepdim=True)
                actor_loss = - I * action_log_prob * advantage - 0.01 * entropy
                actor_loss = actor_loss.mean()
                self.actor.zero_grad()
                actor_loss.backward()
                actor_optim.step()

                ep_return += reward
                done_b |= done
                state = next_state
                I = I * gamma
            
            stats['Actor Loss'].append(actor_loss.item())
            stats['Critic Loss'].append(critic_loss.item())
            stats['Returns'].append(ep_return.mean().item())
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
            action = torch.argmax(self.actor(state_tensor)).item()

            next_state, reward, end = self.next_step(next_state, action)
            frame = self.draw_agent(frame_copy, self.cell_size, next_state)
            self.render(frame, end)  # Render the frame_copy
            self.all_frames.append(frame)


num_envs = os.cpu_count()

env_fns = []

for i in range(num_envs):
    maze = Maze()
    maze.set_up_maze(maze_map_2, False)
    env_fns.append(DeepMazeBaseClass(maze, 2))

parallel_envs = ParallelEnv(env_fns, 500, 500)

parallel_envs.actor_critic(10)
parallel_envs.set_up_maze(maze_map_2, True)
parallel_envs.test_agent((0,0))