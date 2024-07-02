import numpy as np
import cv2 as cv
from maps import maze_map_1, maze_map_2, maze_map_3

actions_mapping = {
    "0": (-1, 0),
    "1": (0, 1),
    "2": (1, 0),
    "3": (0, -1),
}


def override(method):
    return method


class Maze:
    def __init__(self, frame_width=500, frame_height=500) -> None:
        self.empty_maze_frame = None
        self.frame_copy = self.empty_maze_frame
        self.maze_map = None
        self.frame_dim = (frame_width, frame_height)
        self.cell_size = 0
        self.wall_colors = (255, 255, 255)
        self.policy_probs = None
        self.state_values = None
        self.reward_map = None
        self.draw_path = False
        self.all_frames = []

    def env_reset(self):
        return (0,0)

    def policy(self, state):
        return self.policy_probs[state]

    def policy_init(self):
        self.policy_probs = np.full((*self.maze_map.shape, 4), 0.25)

    def state_values_init(self):
        self.state_values = np.full(
            self.maze_map.shape, 0.0
        )  # had to be initially filled with float numbers, otherwise later it would only get updated with integers

    def next_step(self, state, action):
        """
        Return:
            the next state it ends up at, reward, if it terminates
        """
        next_state = state
        action_result = actions_mapping[str(action)]
        next_state_x = max(
            min(self.maze_map.shape[1] - 1, action_result[0] + state[0]), 0
        )
        next_state_y = max(
            min(self.maze_map.shape[0] - 1, action_result[1] + state[1]), 0
        )
        # if the next action yields to a wall, it stays where it is
        if not self.maze_map[next_state_x, next_state_y] == 1:
            next_state = (next_state_x, next_state_y)
        return next_state, self.reward_map[state], self.maze_map[next_state] == 2

    def next_action(self, state):
        possible_actions = self.policy(state)
        prob_best_action = np.max(possible_actions)
        best_possible_actions = np.where(possible_actions == prob_best_action)[0]
        best_action = np.random.choice(best_possible_actions)
        return best_action

    def test_agent(self, state = (0, 0)):
        next_state = state
        end = False
        while not end:
            frame_copy = np.copy(
                self.empty_maze_frame
            )  # Create a new copy for each iteration
            action = self.next_action(next_state)
            next_state, reward, end = self.next_step(next_state, action)
            frame = self.draw_agent(frame_copy, self.cell_size, next_state)
            self.render(frame, end)  # Render the frame_copy
            self.all_frames.append(frame)

    def draw_grid(self, frame, maze_map, frame_dim, cell_size):
        for i in range(0, maze_map.shape[0] + 1):
            cv.line(
                frame,
                (0, i * cell_size),
                (frame_dim[1], i * cell_size),
                self.wall_colors,
                cell_size // 10,
            )
            cv.line(
                frame,
                (i * cell_size, 0),
                (i * cell_size, frame_dim[0]),
                self.wall_colors,
                cell_size // 10,
            )

    def draw_figures(self, frame, maze_map, cell_size):
        tree = cv.imread("figures/tree.jpeg", cv.IMREAD_UNCHANGED)
        tree = cv.resize(tree, (cell_size, cell_size))
        for i in range(maze_map.shape[0]):
            for j in range(maze_map.shape[1]):
                if maze_map[i, j] == 1:
                    top_left_x = j * cell_size
                    bottom_right_x = top_left_x + cell_size
                    top_left_y = i * cell_size
                    bottom_right_y = top_left_y + cell_size
                    frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = tree
        # draw the chest
        coins = cv.imread("figures/coins.jpeg", cv.IMREAD_UNCHANGED)
        coins = cv.resize(coins, (cell_size, cell_size))
        top_left_x = cell_size * (maze_map.shape[0] - 1)
        bottom_right_x = top_left_x + cell_size
        top_left_y = cell_size * (maze_map.shape[1] - 1)
        bottom_right_y = top_left_y + cell_size
        frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = coins

        # draw the house
        house = cv.imread("figures/start.jpeg", cv.IMREAD_UNCHANGED)
        house = cv.resize(house, (cell_size, cell_size))
        top_left_x = 0
        bottom_right_x = top_left_x + cell_size
        top_left_y = 0
        bottom_right_y = top_left_y + cell_size
        frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = house

    def draw_agent(self, frame, cell_size, state=(0, 0)):
        if not self.draw_path:
            frame = np.copy(self.empty_maze_frame)
        agent = cv.imread("figures/agent.jpeg", cv.IMREAD_UNCHANGED)
        agent = cv.resize(agent, (cell_size, cell_size))

        top_left_x = cell_size * state[1]
        bottom_right_x = top_left_x + cell_size
        top_left_y = cell_size * state[0]
        bottom_right_y = top_left_y + cell_size

        frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = agent
        return frame

    def render(self, frame, reached_end):
        cv.imshow("frame", frame)
        cv.waitKey(50)
        if reached_end:
            cv.waitKey(0)
        cv.destroyAllWindows()

    def reward_map_init(self):
        self.reward_map = np.full(self.maze_map.shape, -1)
        self.reward_map[tuple(np.argwhere(self.maze_map == 2)[0])] = 5

    def generate_maze(self, maze_map, frame_width=500, frame_height=500):
        assert frame_width == frame_height
        assert maze_map.shape[0] == maze_map.shape[1]
        assert (
            np.count_nonzero(maze_map == -2) == 1
        )  # make sure starting point is defined
        assert (
            np.count_nonzero(maze_map == 2) == 1
        )  # make sure ending point is also defined

        self.frame_dim = (frame_width, frame_height)
        self.cell_size = frame_width // maze_map.shape[0]
        self.maze_map = maze_map
        self.empty_maze_frame = np.ones((*self.frame_dim, 3), dtype=np.uint8) * 255

        self.draw_grid(
            self.empty_maze_frame, self.maze_map, self.frame_dim, self.cell_size
        )
        self.draw_figures(self.empty_maze_frame, self.maze_map, self.cell_size)

    def create_video_from_frames(self, frames, output_filename, fps=5):
        print("Generating the video...")
        if not frames:
            raise ValueError("The frames list is empty")

        # Get frame size from the first frame
        height, width, layers = frames[0].shape
        size = (width, height)

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        out = cv.VideoWriter(output_filename, fourcc, fps, size)

        for frame in frames:
            out.write(frame)

        out.release()

    def set_up_maze(self, maze_map, draw_the_path):
        self.draw_path = draw_the_path
        self.generate_maze(maze_map, self.frame_dim[0], self.frame_dim[1])
        self.reward_map_init()
        self.policy_init()

    def run_maze(self, maze_map, draw_the_path, output_filename, starting_state):
        self.set_up_maze(maze_map, draw_the_path)
        self.test_agent(starting_state)
        self.create_video_from_frames(self.all_frames, output_filename)


if __name__ == "__main__":
    maze = Maze(1500, 1500)
    maze.run_maze(maze_map_2, False, "output_video.mp4", (0,0))
