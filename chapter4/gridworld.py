import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

# define the actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

#############################
class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=4):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observation is agent's location on grid in with top left corner as (0,0)->0
        # and bottom right is (size-1, size-1) -> size**2-1
        # encoded as an element of {0, ..., `size*size-1`}
        self.observation_space = spaces.Discrete(size*size)

        # We have 4 actions, corresponding to "up", "right", "down", "left"
        self.action_space = spaces.Discrete(4)

        self._target_location = np.array([0, size*size - 1], dtype=int)
        self.nS = self.observation_space.n
        self.nA = self.action_space.n
        self.shape = (self.size, self.size)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "up", 1 to "right" etc.
        """
        self._action_to_direction = {
            UP: np.array([-1, 0]), # move up
            RIGHT: np.array([0, 1]),  # move right
            DOWN: np.array([1, 0]), # move down
            LEFT: np.array([0, -1]), # move left
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._agent_location

    def _get_info(self):
        agent_location_2d = np.unravel_index(self._agent_location, self.shape)
        return {
            "distance": np.min([np.linalg.norm(
                np.subtract(
                    np.unravel_index(target, self.shape),
                    agent_location_2d
                ), 
                ord=1
                ) for target in self._target_location])
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.nS, dtype=int)

        # We will sample the agent's location randomly until it does not coincide with the target location
        while self._terminated():
            self._agent_location = self.np_random.integers(0, self.nS, dtype=int)
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _terminated(self):
        return self._agent_location in self._target_location 

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        agent_loc_curr = self._agent_location 
        agent_loc_curr_2d = np.unravel_index(
            agent_loc_curr, 
            (self.size, self.size)
        )
        
        self._agent_location = np.ravel_multi_index(
            np.clip(
                agent_loc_curr_2d + direction,
                0, 
                self.size - 1
            ), 
            (self.size, self.size)
        )
        # An episode is done iff the agent has reached the target
        terminated = self._terminated()
    
        reward = 0 if agent_loc_curr in self._target_location else -1  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # convert agent and target coordinates from flat index to 2d index
        agent_location_2d = np.unravel_index(
            self._agent_location, 
            self.shape
        )
        target_2d = np.transpose(
            np.unravel_index(
                self._target_location, 
                self.shape
            )
        )

        # First we draw the target
        for location in target_2d:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * location,
                    (pix_square_size, pix_square_size),
                ),
            )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (agent_location_2d + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

#############################

