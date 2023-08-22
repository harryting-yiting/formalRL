import gymnasium as gym
from .. import tools

env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
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
tools.save_frames_as_gif(frames)

