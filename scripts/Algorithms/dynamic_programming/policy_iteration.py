from maze import Maze, override
from maps import maze_map_2
import numpy as np


class MazePolicyIteration(Maze):
    def __init__(self, frame_width=500, frame_height=500) -> None:
        super().__init__(frame_width, frame_height)

    def policy_evaluation(self, policy_probs, state_values, theta=1e-6, gamma=0.99):
        delta = float("inf")

        while delta > theta:
            delta = 0

            for row in range(state_values.shape[0]):
                for col in range(state_values.shape[1]):
                    old_state_value = state_values[(row, col)]
                    new_state_value = 0.0
                    action_probabilities = policy_probs[(row, col)]

                    for action, prob in zip(range(4), action_probabilities):
                        next_state, reward, _ = self.next_step((row, col), action)
                        new_state_value += prob * (
                            reward + gamma * state_values[next_state]
                        )

                    state_values[(row, col)] = new_state_value

                    delta = max(delta, abs(new_state_value - old_state_value))

    def policy_improvement(self, policy_probs, state_values, gamma=0.99):
        policy_stable = True

        for row in range(state_values.shape[0]):
            for col in range(state_values.shape[1]):
                old_action = policy_probs[(row, col)].argmax()
                new_action = None
                max_action_return = float("-inf")

                for action in range(4):
                    next_state, reward, _ = self.next_step((row, col), action)
                    action_return = reward + gamma * state_values[next_state]

                    if action_return > max_action_return:
                        max_action_return = action_return
                        new_action = action

                action_probs = np.zeros(4)
                action_probs[new_action] = 1.0
                policy_probs[(row, col)] = action_probs

                if new_action != old_action:
                    policy_stable = False

        return policy_stable

    def policy_iteration(self, theta=1e-6, gamma=0.99):
        policy_stable = False

        while not policy_stable:
            self.policy_evaluation(self.policy_probs, self.state_values, theta, gamma)
            policy_stable = self.policy_improvement(
                self.policy_probs, self.state_values, gamma
            )

    @override
    def run_maze(self, maze_map, draw_the_path, output_filename):
        self.draw_path = draw_the_path
        self.generate_maze(maze_map, self.frame_dim[0], self.frame_dim[1])
        self.reward_map_init()
        self.policy_init()
        self.state_values_init()
        self.policy_iteration()
        self.test_agent((0, 0))
        self.create_video_from_frames(self.all_frames, output_filename)


if __name__ == "__main__":
    maze = MazePolicyIteration(1500, 1500)
    maze.run_maze(maze_map_2, True, 'output_video.mp4')
