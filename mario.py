import torch
from torch import nn
import torch.functional as F

import numpy as np
import random
import os
import datetime
import copy
from collections import deque


class DoubleDQN(nn.Module):

	def __init__(self, input_dim, num_actions):
		super().__init__()

		c, h, w = input_dim

		if c != 4 : 
			raise ValueError(f"expecting number of input channels to be 3, got c = {c}")

		if h != 84 or w != 84 :
			raise ValueError(f"expecting height and width to be 84, got h = {h} and w = {w}")

		self.online = nn.Sequential(
			nn.Conv2d(c, 32, kernel_size = 8, stride = 4),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size = 3, stride = 1),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(3136, 512),
			nn.ReLU(),
			nn.Linear(512, num_actions))

		self.target = copy.deepcopy(self.online)

		# freeze target net parameters to not get updated via gradient
		for p in self.target.parameters():
			p.requires_grad = False

	def forward(self, input, model):
		if model == "online":
			return self.online(input)
		if model == "target":
			return self.target(input)

















class SuperMario:

	def __init__(self, state_dim, num_actions, device, save_dir):

		self.state_dim = state_dim
		self.num_actions = num_actions
		self.save_dir = save_dir
		self.net = DoubleDQN(self.state_dim, self.num_actions ).to(device)
		self.exploration_rate = 1
		self.exploration_rate_decay = 0.9999
		self.exploration_rate_min = 0.1
		self.curr_step = 0
		self.save_every = 5e5
		self.gamma = 0.9

		self.memory = deque(maxlen = 100000)
		self.batch_size = 32
		self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.00025)
		self.loss_fn = torch.nn.SmoothL1Loss()
		self.device = device

		self.burnin = 1e2 #minimum experiences before training
		self.learn_every = 3 # number of experiences between updates
		self.sync_every = 1e4


	def act(self, state):

		if np.random.rand() < self.exploration_rate :
			action = np.random.randint(self.num_actions)

		else :
			
			state = torch.tensor(state, dtype = torch.float32).to(self.device)
			state = state.unsqueeze(0)
			q_values = self.net(state, model = "online")
			action = torch.argmax(q_values, axis = 1).cpu().numpy()
			action = int(action)

		# decrease exploration rate
		self.exploration_rate *= self.exploration_rate_decay
		self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

		# incerement step
		self.curr_step += 1
		return action


	def cache(self, state, next_state, action, reward, done):

		state = state.__array__()
		next_state = next_state.__array__()

		state = torch.tensor(state).to(self.device)
		next_state = torch.tensor(next_state).to(self.device)
		action = torch.tensor([action]).cuda()
		reward = torch.tensor([reward]).to(self.device)
		done = torch.tensor([done]).to(self.device)

		self.memory.append((state, next_state, action, reward, done))



	def sample(self):

		batch = random.sample(self.memory, self.batch_size)
		state, next_state, action, reward, done = map(torch.stack, zip(*batch))
		return state, next_state, action.squeeze(), reward.squeeze(), done


	def td_estimate(self, state, action):
		q_values = self.net(state, model = "online")[np.arange(0, self.batch_size), action]
		return q_values

	@torch.no_grad()
	def td_target(self, reward, next_state, done):
		next_state_Q = self.net(next_state, model = "online")
		best_action = torch.argmax(next_state_Q, axis = 1)
		next_Q = self.net(next_state, model = "target")[np.arange(0, self.batch_size), best_action]
		return (reward + (1 - done.float()) * self.gamma * next_Q).float()


	def update_Q_online(self,td_estimate, td_target):

		loss = self.loss_fn(td_estimate, td_target)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.item()


	def sync_target_net(self):
		self.target.load_state_dict(self.net.online.state_dict())


	def save(self):
		return 
		# to be continued


	def learn(self):
		print(self.curr_step)

		if self.curr_step % self.sync_every == 0:
			self.sync_target_net

		if self.curr_step % self.save_every == 0:
			self.save()

		if self.curr_step < self.burnin :
			return None, None

		if self.curr_step % self.learn_every != 0:
			return None, None

		state, next_state, action, reward, done = self.sample()

		# getting td estimate
		td_est = self.td_estimate(state, action)

		# getting td target
		td_tgt = self.td_target(reward, next_state, done)

		# backpropagate
		loss = self.update_Q_online(td_est, td_tgt)

		return (td_est.mean().item(), loss)








		