from tqdm import tqdm
import gymnasium as gym
from wrappers import TltlWrapper, PredicateEvaluationResult, PositionPredicate, PositionWrapper, MoveActionWrapper
from gymnasium.spaces.utils import flatten_space, flatten
from gymnasium.spaces import Dict, Discrete
from minigrid.wrappers import DictObservationSpaceWrapper, FullyObsWrapper
from stable_baselines3.common.logger import configure
import numpy as np
import random
"""
Monta Carlo Q-Learning Method 
"""


def initialize_q_table(state_and_action_space):
    qtable = np.zeros(state_and_action_space)
    return qtable


def greedy_policy(qtable, state):
    # Exploitation: take the action with the highest state, action value
    action = np.argmax(qtable[state][:])
    return action


def epsilon_greedy_policy(qtable, state, epsilon):
    # Randomly generate a number between 0 and 1
    random_num = random.uniform(0, 1)
    # if random_num > greater than epsilon --> exploitation
    if random_num > epsilon:
        # Take the action with the highest value given a state
        # np.argmax can be useful here
        action = greedy_policy(qtable, state)
    # else --> exploration
    else:
        action = env.action_space.sample()

    return action


def transform_to_tuple(state):
    return (state['pos'], state['direction'], state['fspa'])


class QLearning:
    """
    Q Learning
    """
    def __init__(self, env):
        self.env = env
        self.q_table = initialize_q_table((env.observation_space['pos'].n, env.observation_space['direction'].n, env.observation_space['fspa'].n, env.action_space.n))
        # Training parameters
        self.n_training_episodes = 150000  # Total training episodes
        self.learning_rate = 0.7  # Learning rate

        # Evaluation parameters
        self.n_eval_episodes = 100  # Total number of test episodes

        # Environment parameters
        self.max_steps = 49  # Max steps per episode
        self.gamma = 0.99  # Discounting rate
        self.eval_seed = []  # The evaluation seed of the environment

        # Exploration parameters
        self.max_epsilon = 1.0  # Exploration probability at start
        self.min_epsilon = 0.05  # Minimum exploration probability
        self.decay_rate = 0.0001  # Exponential decay rate for exploration prob

    def train(self):
        delta = []
        number_sucess = 0
        env.set_random_reset(is_random_reset=True)
        all_rewards = []
        for episode in tqdm(range(self.n_training_episodes)):
            # Reduce epsilon (because we need less and less exploration)
            epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)
            # Reset the environment
            state, info = self.env.reset()
            state = transform_to_tuple(state)
            step = 0
            terminated = False
            truncated = False
            pre_table = np.copy(self.q_table)
            # repeat
            episode_reward = 0
            for step in range(self.max_steps):
                # Choose the action At using epsilon greedy policy
                action = epsilon_greedy_policy(self.q_table, state, epsilon)

                # Take action At and observe Rt+1 and St+1
                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, terminated, truncated, info = env.step(action)
                episode_reward = episode_reward + reward

                new_state = transform_to_tuple(new_state)
                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                self.q_table[state][action] = self.q_table[state][action] + self.learning_rate * (
                            reward + self.gamma * np.max(self.q_table[new_state]) - self.q_table[state][action])

                # If terminated or truncated finish the episode
                if terminated:
                    number_sucess = number_sucess + 1

                if terminated or truncated:
                    break

                # Our next state is the new state
                state = new_state
            delta.append(np.max(np.abs(pre_table - self.q_table)))
            all_rewards.append(episode_reward)
        return self.q_table, delta, number_sucess/self.n_training_episodes, all_rewards

    def evaluate_agent(self, q_table, n_eval_episodes, seed):
        """
        Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
        :param env: The evaluation environment
        :param n_eval_episodes: Number of episode to evaluate the agent
        :param Q: The Q-table
        :param seed: The evaluation seed array (for taxi-v3)
        """
        episode_rewards = []
        success_num = 0
        env.set_random_reset(is_random_reset=False)
        for episode in range(n_eval_episodes):
            if seed:
                state, info = self.env.reset(seed=seed[episode])
            else:
                state, info = self.env.reset()
            state = transform_to_tuple(state)

            step = 0
            truncated = False
            terminated = False
            total_rewards_ep = 0
            print("------------new episod-------------")
            for step in range(self.max_steps):
                # Take the action (index) that have the maximum expected future reward given that state
                action = greedy_policy(q_table, state)
                print(action)
                new_state, reward, terminated, truncated, info = self.env.step(action)
                print(new_state)
                new_state = transform_to_tuple(new_state)
                total_rewards_ep += reward

                if terminated:
                    success_num = success_num + 1
                if terminated or truncated:
                    break
                state = new_state
            episode_rewards.append(total_rewards_ep)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        return mean_reward, std_reward, success_num/n_eval_episodes


if __name__ == "__main__":
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array", max_episode_steps=50)
    # env = TltlWrapper(env, tltl="F p1 & F p2 & F p3 & (! (p1&p3) U p2) & (! p3 U p1)",
    #                   predicates={'p1': PositionPredicate(True, [3, 3]), 'p2': PositionPredicate(True, [5, 5]), 'p3': PositionPredicate(True, [2,2])})

    env = TltlWrapper(env, tltl="F p1 & F p2 & F p3 & (! p1 U p2) & (! p2 U p3)",
                      predicates={'p1': PositionPredicate(True, [3, 3]), 'p2': PositionPredicate(True, [5, 5]),
                                  'p3': PositionPredicate(True, [2, 2])})
    env = PositionWrapper(env)
    env = MoveActionWrapper(env)
    env = gym.wrappers.FilterObservation(env, ["pos", "fspa", "direction"])
    print(env.action_space, env.observation_space)
    print(env.observation_space['pos'].n, env.observation_space['fspa'].n)
    print(env.fspa_all_states)
    q = QLearning(env)
    _, delta, train_success, all_rewards = q.train()
    np.save('./q_table_random_2-5-3_150000', q.q_table)
    np.save('./all_rewards_q_table_random_2-5-3_150000', all_rewards)
    print("train_success: ", train_success)
    mean_reward, std_reward, success_rate = q.evaluate_agent(q.q_table,2, [])
    print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    print("success_rate: ", success_rate)
