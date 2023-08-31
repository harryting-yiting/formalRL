# !/usr/bin/python
import gymnasium as gym
from wrappers import TltlWrapper, PredicateEvaluationResult, PositionPredicate
from gymnasium.spaces.utils import flatten_space, flatten
from minigrid.wrappers import DictObservationSpaceWrapper, FullyObsWrapper
from stable_baselines3.common.logger import configure
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

def make_env(env_id, seed, idx, capture_video, run_name):
	def thunk():
		env = gym.make(env_id, render_mode="rgb_array")
		env = FullyObsWrapper(env)
		env = TltlWrapper(env, tltl="F p1 & F p2 & (! p2 U p1)",
						  predicates={'p1': PositionPredicate(True, [7, 7]), 'p2': PositionPredicate(True, [15, 15])})
		env = gym.wrappers.FilterObservation(env, ["image", "fspa", "direction"])
		env = gym.experimental.wrappers.LambdaObservationV0(env, lambda obs: {**obs, 'image': flatten(
			env.observation_space['image'], obs['image'])}, gym.spaces.Dict(
			{**env.observation_space.spaces, "image": flatten_space(env.observation_space['image'])}))

		env = gym.wrappers.RecordEpisodeStatistics(env)
		if capture_video:
			if idx == 0:
				env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

		env.seed(seed)
		env.action_space.seed(seed)
		env.observation_space.seed(seed)
		return env
	return thunk()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
	torch.nn.init.orthogonal_(layer.weight, std)
	torch.nn.init.constant_(layer.bias, bias_const)
	return layer


class Agent(nn.Module):
	def __init__(self, envs):
		super().__init__()
		self.critic = nn.Sequential(
			layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
			nn.Tanh(),
			layer_init(nn.Linear(64, 64)),
			nn.Tanh(),
			layer_init(nn.Linear(64, 1), std=1.0),
		)
		self.actor = nn.Sequential(
			layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
			nn.Tanh(),
			layer_init(nn.Linear(64, 64)),
			nn.Tanh(),
			layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
		)

	def get_value(self, x):
		return self.critic(x)

	def get_action_and_value(self, x, action=None):
		logits = self.actor(x)
		probs = Categorical(logits=logits)
		if action is None:
			action = probs.sample()
		return action, probs.log_prob(action), probs.entropy(), self.critic(x)

if __name__ == '__main__':
	# init gym
	env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
	env = TltlWrapper(env, tltl="F p1 & F p2 & (! p2 U p1)", predicates={'p1': PositionPredicate(True, [1, 1]), 'p2': PositionPredicate(True, [2, 2])})

	obs, info = env.reset()
	frames = []
	print(env.action_space, env.observation_space)



	# wrap around
	# check obs
	# check action

	# train loop
	# st
	env = gym
