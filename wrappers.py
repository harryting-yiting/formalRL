import numpy as np
import numpy.linalg
from gymnasium.core import Wrapper, ObsType
from fspa import Fspa, State, Predicate, PredicateEvaluationResult
from minigrid import minigrid_env
import numpy as np
from typing import Any
import gymnasium as gym


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
		return PredicateEvaluationResult(distance)


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
		trap = self.fspa.remove_trap_states()
		print(trap)
		print(self.fspa)
		self.fspa.determinize()
		print(self.fspa)
		# wrapper saved variables
		self.current_p = self.fspa.get_init_nodes()[0]  # current fspa state

	def reset(
			self, *, seed: int or None = None, options: dict[str, Any] or None = None
	) -> tuple[ObsType, dict[str, Any]]:
		obs, info = super().reset(seed=seed, options=options)
		self.current_p = self.fspa.get_init_nodes()[0]
		return self.observation(obs), info

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		print('step')
		# compute the reward  = the minimal q given next state
		next_p = self.fspa.next_state_from_mdp_state(self.current_p, MinigridState(self.env))
		reward, edge_index = self.fspa.compute_node_outgoing_without_uc_self(self.current_p, MinigridState(self.env))


		# if next_q is final state
		if next_p in self.fspa.final:
			terminated = True
		# if next_q is trap state
		# elif next_p in self.fspa.trap_state:
		# 	reward = -100
		# 	terminated = True

		self.current_p = next_p
		return self.observation(obs), reward, terminated, truncated, info

	def observation(self, obs):
		return {**obs, "fspa_state": self.current_p}


if __name__ == "__main__":
	# creat env
	env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
	env = TltlWrapper(env, tltl="F p1 & X p2", predicates={'p1': PositionPredicate(True, [1, 1]), 'p2': PositionPredicate(True, [2, 2])})

	observation, info = env.reset(seed=42)
	frames = []
	obs = env.reset()
	for _ in range(1000):
		action = 0
		obs, reward, terminated, truncated, info = env.step(action)
		frames.append(env.render())
		if terminated or truncated:
			observation, info = env.reset()
	env.close()
# TLTLwarpper
# reset
# for
# action
# step
# test q and state and reward
