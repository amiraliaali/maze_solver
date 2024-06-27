from monte_carlo import MazeMonteCarlo
from maze import override
import numpy as np
from maps import maze_map_2


class OffPolicyMonteCarlo(MazeMonteCarlo):
    def __init__(self, frame_width=500, frame_height=500) -> None:
        super().__init__(frame_width, frame_height)

    def target_policy(self, state):
        av = self.action_values[state]
        return np.random.choice(np.flatnonzero(av == av.max()))
    
    def exploratory_policy(self, state, epsilon):
        return self.policy(state, epsilon)
        
    def off_policy_monte_carlo(self, action_values, episodes, gamma=0.99, epsilon=0.2):
        for episode in range(1, episodes+1):
            G = 0
            W = 1
            csa = np.zeros((*self.maze_map.shape, 4))
            state = self.env_reset()
            done = False
            transitions = []

            while not done:
                action = self.exploratory_policy(state, epsilon)
                next_state, reward, done = self.next_step(state, action)
                transitions.append([state, action, reward])
                state = next_state

            for state_t, action_t, reward_t in reversed(transitions):
                G = reward_t + gamma * G
                csa[state_t][action_t] += W
                qsa = action_values[state_t][action_t]

                if csa[state_t][action_t] == 0:
                    continue  # Skip update to prevent division by zero

                delta = (W / csa[state_t][action_t]) * (G - qsa)
                if not np.isnan(delta):
                    action_values[state_t][action_t] += delta

                if action_t != self.target_policy(state_t):
                    continue  # Properly exit the loop if the action is not from the target policy

                W = W * 1. / (1 - epsilon + epsilon / 4)

    @override
    def run_maze(self, maze_map, draw_the_path, output_filename):
        self.draw_path = draw_the_path
        self.generate_maze(maze_map, self.frame_dim[0], self.frame_dim[1])
        self.reward_map_init()
        self.action_values = np.zeros((*self.maze_map.shape, 4))
        self.off_policy_monte_carlo(self.action_values, episodes=5000)
        self.test_agent((0, 0))
        self.create_video_from_frames(self.all_frames, output_filename)


if __name__ == "__main__":
    maze = OffPolicyMonteCarlo(1500, 1500)
    maze.run_maze(maze_map_2, True, "output_video.mp4")
