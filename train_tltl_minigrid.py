# !/usr/bin/python

from wrappers import TltlWrapper, PositionPredicate
import gymnasium as gym
import numpy as np


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
