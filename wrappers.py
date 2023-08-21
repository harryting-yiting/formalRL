from gymnasium.core import Wrapper
from fspa import Fspa


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

	def __init__(self, tltl: str, predicates: dict):
		self.fspa = Fspa(predicates=predicates)
		self.fspa.from_formula(tltl)
		self.current_p = self.fspa.  # current fspa state

	def step(self, action):

