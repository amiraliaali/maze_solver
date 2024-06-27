from maze import Maze, override
import numpy as np


class MazeTemporalDifference(Maze):
    def __init__(self, frame_width=500, frame_height=500) -> None:
        super().__init__(frame_width, frame_height)

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