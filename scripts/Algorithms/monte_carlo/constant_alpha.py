from monte_carlo import MazeMonteCarlo
from maze import override
import numpy as np
from maps import maze_map_2


class OnPolicyConstantAlphaMonteCarlo(MazeMonteCarlo):
    def __init__(self, frame_width=500, frame_height=500) -> None:
        super().__init__(frame_width, frame_height)
        
    def on_policy_constant_alpha_monte_carlo(self, action_values, episodes, gamma=0.99, epsilon=0.2, alpha=0.1):
        for episode in range(1, episodes+1):
            state = self.env_reset()
            done = False
            transitions = []
            while not done:
                action = self.policy(state, epsilon)
                next_state, reward, done = self.next_step(state, action)
                transitions.append([state, action, reward])
                state = next_state
            G = 0
            for state_t, action_t, reward_t in reversed(transitions):
                G = reward_t + gamma*G
                action_values[state_t][action_t] += alpha * (G - self.action_values[state_t][action_t])

    @override
    def run_maze(self, maze_map, draw_the_path, output_filename):
        self.draw_path = draw_the_path
        self.generate_maze(maze_map, self.frame_dim[0], self.frame_dim[1])
        self.reward_map_init()
        self.action_values = np.zeros((*self.maze_map.shape, 4))
        self.on_policy_constant_alpha_monte_carlo(self.action_values, episodes=200)
        self.test_agent((0, 0))
        self.create_video_from_frames(self.all_frames, output_filename)


if __name__ == "__main__":
    maze = OnPolicyConstantAlphaMonteCarlo(1500, 1500)
    maze.run_maze(maze_map_2, True, "output_video.mp4")
