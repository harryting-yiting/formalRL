from tqdm import tqdm
import gymnasium as gym
from wrappers import TltlWrapper, PredicateEvaluationResult, PositionPredicate, PositionWrapper, MoveActionWrapper
from gymnasium.spaces.utils import flatten_space, flatten
from gymnasium.spaces import Dict, Discrete
from minigrid.wrappers import DictObservationSpaceWrapper, FullyObsWrapper
from stable_baselines3.common.logger import configure
import numpy as np
import random
import matplotlib.pyplot as plt

"""
    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |
"""

if __name__ == "__main__":
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array", max_episode_steps=50)
    env = TltlWrapper(env, tltl="F p1 & F p2 & (! p2 U p1)",
                      predicates={'p1': PositionPredicate(True, [3, 3]), 'p2': PositionPredicate(True, [5, 5])})
    env = PositionWrapper(env)
    env = MoveActionWrapper(env)
    env = gym.wrappers.FilterObservation(env, ["pos", "fspa", "direction"])

    state, info = env.reset()
    print(env.fspa_all_states)
    print(state)
    print(env.agent_pos)


    env.place_agent((3,2), (1,1), False)
    action = 1
    state, reward, terminated, truncated, info =  env.step(action)
    print('__________________________________________________')
    print(state)
    print(env.agent_pos)
    print("term, trunc: ", terminated, truncated, reward)


    action = 2
    state, reward, terminated, truncated, info = env.step(action)
    print('__________________________________________________')
    print(state)
    print(env.agent_pos)
    print("term, trunc: ", terminated, truncated, reward)


    action = 0
    state, reward, terminated, truncated, info = env.step(action)
    print('__________________________________________________')
    print(state)
    print(env.agent_pos)
    print("term, trunc: ", terminated, truncated, reward)


    env.place_agent((5,4), (1,1), False)
    action = 1
    state, reward, terminated, truncated, info =  env.step(action)
    print('__________________________________________________')
    print(state)
    print(env.agent_pos)
    print("term, trunc: ", terminated, truncated, reward)

    action = 2
    state, reward, terminated, truncated, info = env.step(action)
    print('__________________________________________________')
    print(state)
    print(env.agent_pos)
    print("term, trunc: ", terminated, truncated, reward)

    frame = env.render()
    plt.imshow(frame)
    plt.show()