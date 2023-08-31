import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
from gymnasium.core import Wrapper, ObsType, ObservationWrapper
from fspa import Fspa, State, Predicate, PredicateEvaluationResult
from minigrid import minigrid_env
import numpy as np
from typing import Any
import gymnasium as gym
from tools import save_frames_as_gif

class MinigridState(State):
	def __init__(self, state: minigrid_env):
		self.state = state

	def get_state(self) -> minigrid_env:
		return self.state


# TLTL for a specific env
# empty env
# input: predicates{'a': function,'b': function}
# define a set of predicates


## assumption:
## env is fully observable
## current position is known
## target position is known

class PositionPredicate(Predicate):
	def __init__(self, actionable: bool, target_position):
		super().__init__(actionable=actionable)
		self.target_ps = np.array(target_position)

	def evaluate(self, s: MinigridState) -> PredicateEvaluationResult:
		env: minigrid_env = s.get_state()  # type: minigrid_env
		current_position = np.array(env.agent_pos)
		distance = numpy.linalg.norm(self.target_ps - current_position)
		return PredicateEvaluationResult(0.2 - distance)


class TltlWrapper(Wrapper):
	"""
	Wrapper to compute the fspa state and reward.
	This can be used to control the RL agent to follow TLTL specifications.
	The returned state is [fspa state, mdp state]

	Example:
		# >>> import gymnasium as gym
		# >>> from wrappers import TltlWrapper
		# >>>
	"""

	def __init__(self, env, tltl: str, predicates: dict[str, Predicate]):
		super().__init__(env)
		self.fspa = Fspa(predicates=predicates)
		self.fspa.from_formula(tltl)
		# wrapper saved variables
		self.current_p = self.fspa.get_init_node()  # current fspa state
		self.fspa_all_states = list(self.fspa.get_all_states())
		new_fspa_state = gym.spaces.Discrete(len(self.fspa_all_states))
		self.observation_space = gym.spaces.Dict({**self.observation_space.spaces, "fspa": new_fspa_state})

	def reset(
			self, *, seed: int or None = None, options: dict[str, Any] or None = None
	) -> tuple[ObsType, dict[str, Any]]:
		obs, info = super().reset(seed=seed, options=options)

		# count the first MDP state
		self.current_p = self.fspa.get_init_node()
		# random choosing begin p
		# self.current_p = self.fspa_all_states[self.observation_space['fspa'].sample()]
		
		# self.env.place_agent(rand_dir=True)
		self.fspa.update_out_edge_predicates(self.current_p, MinigridState(self.env))
		self.current_p = self.fspa.get_next_state_from_mdp_state(self.current_p)
		return self.observation(obs), info

	def step(self, action):
		obs, _, _, truncated, info = self.env.step(action)
		# compute the reward  = the minimal q given next state
		self.fspa.update_out_edge_predicates(self.current_p, MinigridState(self.env))
		reward = self.fspa.get_reward(self.current_p)
		if(reward > 0):
			reward = 50
		else:
			reward = 0
			
		next_p = self.fspa.get_next_state_from_mdp_state(self.current_p)
		# if next_q is final state

		terminated = (next_p in self.fspa.final)
		if terminated:
			# print("terminated")
			# print(terminated,"next p", next_p, self.env.agent_pos, reward)
			reward = 100
			reward = reward - 50 * (self.step_count/self.max_steps)
			#reward = reward - 0.9 * (self.step_count/self.max_steps)
		if next_p in self.fspa.trap:
			truncated = True
		
		if truncated:
			reward = -100

		self.current_p = next_p
		
		return self.observation(obs), reward, terminated, truncated, info

	def observation(self, obs):
		return {**obs, "fspa": self.fspa_all_states.index(self.current_p)}



if __name__ == "__main__":
	# creat env
	env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array", max_steps=20)
	env = TltlWrapper(env, tltl="F p1 & F p2 & (! p2 U p1)", predicates={'p1': PositionPredicate(True, [3, 3]), 'p2': PositionPredicate(True, [6, 6])})
	print(env.fspa)
	frames = []
	obs, info = env.reset()
	print("Init FSPA", obs["fspa"])
	print("Init Position", env.agent_pos)
	action = [2,2,1,2,2,2,2,2,0,2,2,2,2]
	for i in range(0, len(action)):
		frames.append(env.render())
		obs, reward, terminated, truncated, info = env.step(action[i])
		print('____________________________________')
		print("fspa state: ", obs["fspa"])
		print("Agent Position", env.agent_pos)
		print('rewar', reward)

		if terminated or truncated:
			observation, info = env.reset()
	env.close()
	save_frames_as_gif(frames)
# TLTLwarpper
# reset
# for
# action
# step
# test q and state and reward
