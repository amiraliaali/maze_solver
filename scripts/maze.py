import numpy as np
import cv2 as cv
from maps import maze_map_1

actions_mapping = {
    "0": (-1, 0),
    "1": (0, 1),
    "2": (1, 0),
    "3": (0, -1),
}

def override(method):
    return method

class Maze:
    def __init__(self) -> None:
        self.empty_maze_frame = None
        self.frame_copy = self.empty_maze_frame
        self.maze_map = None
        self.frame_dim = (0, 0)
        self.cell_size = 0
        self.wall_colors = (100, 100, 100)
        self.agent_color = (255, 50, 50)
        self.policy_probs = None
        self.state_values = None
        self.reward_map = None
        self.draw_path = False

    def policy(self, state):
        return self.policy_probs[state]

    def policy_init(self):
        self.policy_probs = np.full((*self.maze_map.shape, 4), 0.25)


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
        return next_state, self.reward_map[state], self.maze_map[state] == 2

    def next_action(self, state):
        possible_actions = self.policy(state)
        prob_best_action = np.max(possible_actions)
        best_possible_actions = np.where(possible_actions == prob_best_action)[0]
        best_action = np.random.choice(best_possible_actions)
        return best_action

    def policy_evaluation(self, gamma=0.99, theta=1e-6):
        while True:
            delta = 0
            for row in range(self.maze_map.shape[0]):
                for col in range(self.maze_map.shape[1]):
                    v = self.state_values[(row, col)]
                    new_value = 0
                    for action in range(4):
                        next_state, reward, _ = self.next_step((row, col), action)
                        new_value += self.policy_probs[(row, col, action)] * (
                            reward + gamma * self.state_values[next_state]
                        )
                    self.state_values[(row, col)] = new_value
                    delta = max(delta, abs(v - new_value))
            if delta < theta:
                break

    def policy_improvement(self, gamma=0.99):
        policy_stable = True
        for row in range(self.maze_map.shape[0]):
            for col in range(self.maze_map.shape[1]):
                old_action = np.argmax(self.policy_probs[(row, col)])
                action_values = np.zeros(4)
                for action in range(4):
                    next_state, reward, _ = self.next_step((row, col), action)
                    action_values[action] = (
                        reward + gamma * self.state_values[next_state]
                    )
                best_action = np.argmax(action_values)
                if old_action != best_action:
                    policy_stable = False
                self.policy_probs[(row, col)] = np.eye(4)[best_action]
        return policy_stable

    def policy_iteration(self, gamma=0.99, theta=0.5):
        policy_stable = False
        while not policy_stable:
            self.policy_evaluation(gamma, theta)
            policy_stable = self.policy_improvement(gamma)

    def test_agent(self, state):
        self.frame_copy = np.copy(self.empty_maze_frame)
        next_state = state
        end = False
        while not end:
            action = self.next_action(next_state)
            next_state, reward, end = self.next_step(next_state, action)
            frame = self.draw_agent(self.frame_copy, self.cell_size, next_state)
            self.render(frame, end)

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

    def draw_walls_and_goal(self, frame, maze_map, cell_size):
        for i in range(maze_map.shape[0]):
            for j in range(maze_map.shape[1]):
                if maze_map[i, j] == 1:
                    cv.rectangle(
                        frame,
                        (j * cell_size, i * cell_size),
                        ((j + 1) * cell_size, (i + 1) * cell_size),
                        self.wall_colors,
                        -1,
                    )
                elif maze_map[i, j] == 2:
                    cv.rectangle(
                        frame,
                        (j * cell_size, i * cell_size),
                        ((j + 1) * cell_size, (i + 1) * cell_size),
                        (50, 50, 255),
                        -1,
                    )
                elif maze_map[i, j] == -2:
                    cv.rectangle(
                        frame,
                        (j * cell_size, i * cell_size),
                        ((j + 1) * cell_size, (i + 1) * cell_size),
                        (50, 255, 50),
                        -1,
                    )

    def draw_agent(self, frame, cell_size, state=(0, 0)):
        if not self.draw_path:
            frame = np.copy(self.empty_maze_frame)
        centre_x = cell_size * state[1] + cell_size // 2
        centre_y = cell_size * state[0] + cell_size // 2
        cv.circle(frame, (centre_x, centre_y), cell_size // 3, self.agent_color, -1)
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
        self.draw_walls_and_goal(self.empty_maze_frame, self.maze_map, self.cell_size)
    
    def run_maze(self, maze_map, draw_the_path, frame_width=500, frame_height=500):
        self.draw_path = draw_the_path
        self.generate_maze(maze_map, frame_width, frame_height)
        self.reward_map_init()
        self.policy_init()
        self.test_agent((0, 0))

if __name__ == "__main__":
    maze = Maze()
    maze.run_maze(maze_map_1, False, 1500, 1500)
