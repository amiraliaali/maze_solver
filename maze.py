import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

maze_map = np.array([
    [-2, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 0, 2],
])

actions_mapping = {
    "0" : (-1, 0),
    "1" : (0, 1),
    "2" : (1, 0),
    "3" : (0, -1),
}

class Maze:
    def __init__(self) -> None:
        self.empty_maze_frame = None
        self.frame_copy = self.empty_maze_frame
        self.maze_map = None
        self.frame_dim = (0, 0)
        self.cell_size = 0
        self.wall_colors = (100, 255, 100)
        self.agent_color = (255, 100, 100)
        self.policy_probs = None
        self.state=(0,0)

    def policy(self, state):
        return self.policy_probs[state]
    
    def policy_init(self):
        self.policy_probs = np.full((*self.maze_map.shape, 4), 0.25)

    def next_step(self, state):
        possible_actions = self.policy(state)
        prob_best_action = np.max(possible_actions)
        best_possible_actions = np.where(possible_actions == prob_best_action)[0]
        best_action = np.random.choice(best_possible_actions)
        action_result = actions_mapping[str(best_action)]
        next_state_x = max(min(maze_map.shape[1]-1, action_result[0] + state[0]), 0)
        next_state_y = max(min(maze_map.shape[0]-1, action_result[1] + state[1]), 0)
        if not maze_map[next_state_x, next_state_y] == 1:
            state = (next_state_x, next_state_y)
        return state, self.maze_map[state] == 2

    def test_agent(self, state):
        self.frame_copy = np.copy(self.empty_maze_frame)
        next_state = state
        end = False
        while not end:
            next_state, end = self.next_step(next_state)
            frame = self.draw_agent(self.cell_size, next_state)
            self.render(frame)
            print(f"next_state= {next_state}")


    def draw_grid(self, frame, maze_map, frame_dim, cell_size):
        for i in range(0, maze_map.shape[0]+1):
            cv.line(frame, (0, i*cell_size), (frame_dim[1], i*cell_size), self.wall_colors, cell_size//10 )
            cv.line(frame, (i*cell_size, 0), (i*cell_size, frame_dim[0]), self.wall_colors, cell_size//10 )

    def draw_walls(self, frame, maze_map, cell_size):
        for i in range(maze_map.shape[0]):
            for j in range(maze_map.shape[1]):
                if maze_map[i, j] == 1:
                    cv.rectangle(frame, (j*cell_size, i*cell_size), ((j+1)*cell_size, (i+1)*cell_size), self.wall_colors, -1)

    def draw_agent(self, cell_size, state=(0, 0)):
        frame = np.copy(self.empty_maze_frame)
        centre_x = cell_size*state[1] + cell_size//2
        centre_y = cell_size*state[0] + cell_size//2
        cv.circle(frame, (centre_x, centre_y), cell_size//4, self.agent_color, -1)
        return frame

    def render(self, frame):
        cv.imshow("frame", frame)
        cv.waitKey(25)
        cv.destroyAllWindows()


    def generate_maze(self, maze_map, frame_width=500, frame_height=500):
        assert frame_width == frame_height
        assert maze_map.shape[0] == maze_map.shape[1]
        assert np.count_nonzero(maze_map == -2) == 1 # make sure starting point is defined
        assert np.count_nonzero(maze_map == 2) == 1 # make sure ending point is also defined

        self.frame_dim = (frame_width, frame_height)
        self.cell_size = frame_width // maze_map.shape[0]
        self.maze_map = maze_map
        self.empty_maze_frame = np.zeros((*self.frame_dim, 3), dtype=np.uint8)

        self.draw_grid(self.empty_maze_frame, self.maze_map, self.frame_dim, self.cell_size)
        self.draw_walls(self.empty_maze_frame, self.maze_map, self.cell_size)
        self.policy_init()
        self.test_agent((0,0))
        

maze = Maze()
maze.generate_maze(maze_map, 1500, 1500)



