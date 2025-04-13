import numpy as np
from rlang.grounding.utils.primitives  import VectorState
from tqdm import tqdm
import gymnasium as gym
import rlang
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import pygame
import os
import sys
from utils import plot_training_rewards, plot_comparison_training_rewards
three_folders_up = os.path.abspath(os.path.join(__file__, f"../../agents/"))
print(three_folders_up)
sys.path.append(three_folders_up)
from base_r_max import BaseRLangRMaxAgent

class RLangRmaxAgent(BaseRLangRMaxAgent):
    def __init__(self, env, knowledge=None, num_states=500, num_actions=6, r_max=20, gamma=0.95, delta=0.01, M=1):
        super().__init__(env, knowledge=knowledge, r_max=r_max, gamma=gamma, delta=delta, M=M)
        self.num_states = num_states
        self.num_actions = num_actions

    def state_to_vector(self, state):
        width = self.env.unwrapped.desc.shape[1]
        return [state % width, state // width]
    
    def preload_knowledge(self):
        if not self.knowledge:
            return

        for state in range(self.num_states):
            for action in range(self.num_actions):
                try:
                    vector_state = VectorState(self.state_to_vector(state))
                    reward = int(self.knowledge.reward_function(state=vector_state, action=action)[0])
                    next_state_dist = self.knowledge.transition_function(state=vector_state, action=action)
                except AttributeError as e:
                    #print(f"Error in knowledge function: {e}")
                    reward = self.r_max
                    next_state_dist = {}
                self.emp_reward_dist[state, action] = reward
                for next_state, prob in next_state_dist.items():
                    row, col = next_state
                    nrows, ncols = self.env.unwrapped.desc.shape
                    if not (0 <= row < nrows and 0 <= col < ncols):
                        continue  # skip invalid symbolic prediction
                    encoded_state = row * ncols + col
                    self.emp_transition_dist[action, state, encoded_state] = prob
        assert np.all(self.emp_transition_dist >= 0) and np.all(self.emp_transition_dist <= 1) # Ensures probability distribution is valid

    def test(self, num_episodes=10,render=True):
        self.env = gym.make(self.env.spec.id, desc=None, map_name="4x4", render_mode='human' if render else 'rgb_array', is_slippery=False)

        total_rewards = []  
      
        for episode in range(num_episodes):
            state, info = self.env.reset()  
            total_reward = 0
            q_optimal = self.compute_near_optimal_value_function()
            terminated=False
            truncated=False
    
            while not terminated and not truncated:

                self.env.render()  
                action = self.select_action(q_optimal, state)  
                new_state, reward, terminated, truncated, _ = self.env.step(action)

                total_reward += reward
                state = new_state

            total_rewards.append(total_reward)

        if render:
            self.env.close()
            self.env = gym.make(self.env.spec.id)
            self.env.reset()
        
        self.env.close()
  

        return np.mean(total_rewards)



if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)
    knowledge = rlang.parse_file("./frozen_lake.rlang")    
    agent_with_policy = RLangRmaxAgent(env,knowledge=knowledge,num_states=16,num_actions=4, gamma=0.9, r_max=100)
    rewards_with_policy = agent_with_policy.train(episodes=500)
    print(f"Average reward with policy: {agent_with_policy.test(10, render=False)}")
    agent = RLangRmaxAgent(env,num_states=16,num_actions=4)
    rewards = agent.train(episodes=500)
    print(f"Average reward without policy: {agent.test(10, render=False)}")
    plot_training_rewards(rewards_with_policy,save_path="./plots/rmax_training_rewards_knowledge.png")
    plot_training_rewards(rewards,save_path="./plots/rmax_training_rewards.png")
    plot_comparison_training_rewards(
        reward_dict={
            "With RLang Policy": rewards_with_policy,
            "Without RLang": rewards
        },
        save_path="./plots/rmax_learning_comparison.png"
    )




