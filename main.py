from env import get_super_mario_env
from mario import DoubleDQN
from mario import SuperMario

import torch





if torch.cuda.is_available():
	device = torch.device("cuda")
save_dir = ""

env = get_super_mario_env()
env.reset()
mario = SuperMario(state_dim = (4,84,84), num_actions = env.action_space.n, save_dir = save_dir, device = device)
episodes = 1000000000000

for e in range(episodes):

	print("e = " + str(e))

	state = env.reset()
	while True :
		state = state.__array__()
		action = mario.act(state)

		next_state, reward, done, info = env.step(action)
		mario.cache(state, next_state, action, reward, done)
		q, loss = mario.learn()

		state = next_state
		if done or info["flag_get"]:
			break
