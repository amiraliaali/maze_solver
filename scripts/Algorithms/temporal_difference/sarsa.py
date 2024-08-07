from temporal_difference import MazeTemporalDifference
import numpy as np
from maps import maze_map_4
from maze import override

class MazeSarsa(MazeTemporalDifference):
    def sarsa(self, action_values, episodes, alpha=0.1, gamma=0.99, epsilon=0.2):
        for episode in range(1, episodes+1):
            state = self.env_reset()
            action = self.policy(state, epsilon)
            done = False

            while not done:
                next_state, reward, done = self.next_step(state, action)
                next_action = self.policy(next_state, epsilon)

                qsa = action_values[state][action]
                next_qsa = action_values[next_state][next_action]

                action_values[state][action] = qsa + alpha*(reward + gamma*next_qsa - qsa)

                state = next_state
                action = next_action

    @override
    def run_maze(self, maze_map, draw_the_path, output_filename):
        self.draw_path = draw_the_path
        self.generate_maze(maze_map, self.frame_dim[0], self.frame_dim[1])
        self.reward_map_init()
        self.action_values = np.zeros((*self.maze_map.shape, 4))
        self.sarsa(self.action_values, episodes=200)
        self.test_agent((0, 0))
        self.create_video_from_frames(self.all_frames, output_filename)


if __name__ == "__main__":
    maze = MazeSarsa(1500, 1500)
    maze.run_maze(maze_map_4, True, "output_video.mp4")
