# Import environment base class from OpenAI Gym
import keyboard
from gym import Env
import math
# Import gym spaces
import time
from gym.spaces import Discrete, Box
from VizHalo.Halohelper import Halogame
# Import opencv
import cv2
import numpy as np

class HaloGym(Env):
    # Function that is called when we start the env
    def __init__(self, render=False):
        # Inherit from Env
        super().__init__()
        # Setup the game
        self.game = Halogame()
        # self.game.load_config('github/VizDoom/scenarios/basic.cfg')
        # Create the action space and observation space
        self.observation_space = Box(low=0, high=255, shape=(160, 100))
        self.action_space = Discrete(11)

    # This is how we take a step in the environment
    def step(self, action):
        # Specify action and take step
        actions = np.identity(11)
        # reward = self.game.make_action(actions[action])
        print(self.game.get_state())
        # Get all the other stuff
        # we need to return
        if self.game.get_state():
            reward_movement, reward_shoot = self.game.make_action(actions[action])
            reward_health, reward_buffer, reward_ammo = self.game.set_additional_reward()
            reward = math.fsum([reward_health,reward_buffer*3,reward_ammo, reward_movement, reward_shoot])
            state = cv2.resize(cv2.cvtColor(self.game.get_state().screen_buffer, cv2.COLOR_BGR2GRAY),(160,100)).reshape(160,100)
            print(self.observation_space.shape == state.shape)
            ammo = self.game.get_state().get_ammo()

            info = ammo
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0

        info = {"info": info}
        done = bool(1-reward_health) or (reward_buffer >= 0.74)
        print(reward_health, reward_buffer)
        return state, reward, done, info

        # Define how to render the game or environment

    def render(self):
        cv2.imshow('window',cv2.resize(cv2.cvtColor(self.game.get_state().screen_buffer, cv2.COLOR_BGR2GRAY), (100, 160)))

    # What happens when we start a new game
    def reset(self):
        self.game.new_episode()
        state = cv2.resize(cv2.cvtColor(self.game.get_state().screen_buffer, cv2.COLOR_BGR2GRAY),(100,160)).reshape(160,100).astype('int32')
        return state
    # Call to close
    # down the game
    def close(self):
        print('just before closed')
        self.game.close()
