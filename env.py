import warnings
warnings.filterwarnings('ignore')

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T

import numpy as np
from PIL import Image
import random
from collections import deque

import gym
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
# https://github.com/Kautenja/nes-py


env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
# actions are limited to 0. walk right and 1. jump right
env = JoypadSpace(env, [["right"], ["right", "A"]])
# moving right is first action
# A for jumping , so jumping right is second action
env.reset()

next_state, reward, done, info = env.step(action = 0)


class SkipFrame(gym.Wrapper) :

	def __init__(self, env, skip):
		super().__init__(env)
		self._skip = skip # int

	def step(self, action):

		total_reward = 0
		done = False
		for i in range(self._skip):
			state, reward, done, info = self.env.step(action = action)
			total_reward += reward
			if done :
				break

		# return last observed state
		return state, total_reward, done, info

class ObservationWrapper(gym.ObservationWrapper):

	def __init__(self, env, shape):
		super().__init__(env)

		if isinstance(shape, int):
			self.shape = (shape, shape)
		else :
			self.shape = tuple(shape)

		self.transform = T.Compose([ T.Grayscale(), T.Resize(self.shape), T.Normalize(0, 255)])



	def observation(self, observation):
		observation = torch.tensor(observation.copy(), dtype = torch.float32)
		observation = observation.permute((2,0,1))
		observation = self.transform(observation)
		observation = observation.squeeze(0)
		return observation


env = SkipFrame(env, skip=4)
env = ObservationWrapper(env, shape=84)
env = FrameStack(env, num_stack=4)

def get_super_mario_env():
	return env