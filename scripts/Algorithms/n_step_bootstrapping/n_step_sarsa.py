from maze import Maze, override
import numpy as np
from maps import maze_map_4

class MazeNStepSarsa(Maze):
    @override
    def policy(self, state, epsilon=0.):
        if np.random.random() < epsilon:
            return np.random.randint(4)
        else:
            av = self.action_values[state]
            return np.random.choice(np.flatnonzero(av == av.max()))
        
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
        self.n_step_sarsa(self.action_values, episodes=200)
        self.test_agent((0, 0))
        self.create_video_from_frames(self.all_frames, output_filename)
        
    def n_step_sarsa(self, action_values, episodes, alpha=0.1, gamma=0.99, epsilon=0.2, n=8):
        for episode in range(1, episodes+1):
            state = self.env_reset()
            action = self.policy(state, epsilon)
            transitions = []
            done = False
            t = 0

            while t-n < len(transitions):
                if not done:
                    next_state, reward, done = self.next_step(state, action)
                    next_action = self.policy(next_state, epsilon)
                    transitions.append([state, action, reward])

                if t >= n:
                    G = action_values[next_state][next_action]

                    for state_t, action_t, reward_t in reversed(transitions[t-n:]):
                        G = reward_t + gamma * G
                    action_values[state][action] += alpha *(G - action_values[state_t][action_t])

                t += 1
                state = next_state
                action = next_action


if __name__ == "__main__":
    maze = MazeNStepSarsa(1500, 1500)
    maze.run_maze(maze_map_4, True, "output_video.mp4")
