from maze import Maze, override
from maps import maze_map_2
import numpy as np


class MazeMonteCarlo(Maze):
    def __init__(self, frame_width=500, frame_height=500) -> None:
        super().__init__(frame_width, frame_height)

    @override
    def policy(self, state, epsilon=0.):
        if np.random.random() < epsilon:
            return np.random.randint(4)
        else:
            av = self.action_values[state]
            return np.random.choice(np.flatnonzero(av == av.max()))
        
    def on_policy_monte_carlo(self, action_values, episodes, gamma=0.99, epsilon=0.2):
        sa_returns = {}
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
                if not (state_t, action_t) in sa_returns:
                    sa_returns[(state_t, action_t)] = []
                
                sa_returns[(state_t, action_t)].append(G)
                action_values[state_t][action_t] = np.mean(sa_returns[((state_t, action_t))])

    @override
    def test_agent(self, state = (0, 0)):
        next_state = state
        end = False
        while not end:
            frame_copy = np.copy(
                self.empty_maze_frame
            )  # Create a new copy for each iteration
            action = np.argmax(self.action_values[next_state])
            next_state, reward, end = self.next_step(next_state, action)
            frame = self.draw_agent(frame_copy, self.cell_size, next_state)
            self.render(frame, end)  # Render the frame_copy
            self.all_frames.append(frame)

    @override
    def run_maze(self, maze_map, draw_the_path, output_filename):
        self.draw_path = draw_the_path
        self.generate_maze(maze_map, self.frame_dim[0], self.frame_dim[1])
        self.reward_map_init()
        self.action_values = np.zeros((*self.maze_map.shape, 4))
        self.on_policy_monte_carlo(self.action_values, episodes=200)
        self.test_agent((0, 0))
        self.create_video_from_frames(self.all_frames, output_filename)


if __name__ == "__main__":
    maze = MazeMonteCarlo(1500, 1500)
    maze.run_maze(maze_map_2, True, "output_video.mp4")
