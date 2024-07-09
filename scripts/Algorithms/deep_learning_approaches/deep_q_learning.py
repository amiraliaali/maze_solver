from maze import Maze, override
from base_class import ReplayMemory, DeepMazeBaseClass
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.optim import AdamW
from maps import maze_map_3
from tqdm import tqdm


class DeepQLearning(DeepMazeBaseClass):
    def __init__(self, original_maze, state_dim, frame_width=500, frame_height=500) -> None:
        super().__init__(original_maze, state_dim, frame_width, frame_height)
    
    def train_deep_q_learning(self, episodes, alpha=0.001, batch_size=32, gamma=0.99, epsilon=0.05):
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

                    qsa_b = self.q_network(state_b).gather(1, action_b) # takes the q value of the action which was taken

                    next_qsa_b = self.target_q_network(next_state_b)
                    next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0] # takes the q value of the optimal action

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
    def run_maze(self, maze_map, draw_the_path, output_filename):
        self.set_up_maze(maze_map, draw_the_path)
        self.test_agent((0,0))
        self.create_video_from_frames(self.all_frames, output_filename)


if __name__ == "__main__":
    maze = Maze()
    maze.set_up_maze(maze_map_3, False)
    maze = DeepQLearning(maze, 2)
    maze.train_deep_q_learning(episodes=200)
    maze.run_maze(maze_map_3, False, "output_video.mp4")
