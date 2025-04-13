import numpy as np
import gymnasium as gym
import random
from collections import defaultdict
from tqdm import tqdm
import rlang
from rlang.grounding.utils.primitives import VectorState
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import pygame
from utils import plot_training_rewards, plot_comparison_training_rewards
import os
import sys
three_folders_up = os.path.abspath(os.path.join(__file__, f"../../agents/"))
print(three_folders_up)
sys.path.append(three_folders_up)
from base_dyna_q import BaseDynaQAgent

class RLangDynaQAgent(BaseDynaQAgent):
    def __init__(self, env, n_planning_steps, alpha=0.1, gamma=0.99, epsilon=0, knowledge=None,policy_name=None, p_policy=0.2):
       super().__init__(env, n_planning_steps, alpha=0.1, gamma=0.99, epsilon=0.1, knowledge=knowledge,policy_name=policy_name, p_policy=0.2)
        
    
    def state_to_vector(self, state):
        width = self.env.unwrapped.desc.shape[1]
        return [state % width, state // width]
    
    def preload_knowledge(self):
        q_func = defaultdict(lambda: defaultdict(lambda: 0))
        reward_function = self.knowledge.reward_function
        transition_function = self.knowledge.transition_function
        states = range(self.env.observation_space.n)
        actions = range(self.env.action_space.n)
        
        if reward_function:
            for s in states:
                vector_s = self.state_to_vector(s)
                for i, a in enumerate(actions):
                    q_func[s][a] = reward_function(state=VectorState(vector_s), action=i)

        if transition_function and reward_function:
            for s in states:
                vector_s = self.state_to_vector(s)
                for a in actions:
                    s_prime_dist = transition_function(state=VectorState(vector_s), action=a)
                    if s_prime_dist:
                        r_prime = self.weighted_reward(reward_function, s_prime_dist, action=a)
                        v_s_prime = self.weighted_value(q_func, s_prime_dist, actions)
                        q_func[s][a] += self.alpha * (r_prime + self.gamma * v_s_prime)
                        
        return defaultdict(lambda: np.zeros(self.env.action_space.n), {
            state: np.array([q_func[state][a] for a in actions]) for state in states
        })
    def select_action(self, state):
            if self.knowledge and random.random() < self.p_policy:
                vector_s = self.state_to_vector(state)
                action = self.policy(state=VectorState(vector_s))
                #print(f"Policy output for state {vector_s}: {action}")
                return int(list(action.keys())[0][0])
            
            if random.random() < self.epsilon:
                return self.env.action_space.sample()
            return np.argmax(self.q_table[state])
    
    def test(self, num_episodes=100,render=True):
        self.env = gym.make(self.env.spec.id, desc=None, map_name="4x4", is_slippery=False, render_mode="human" if render else "rgb_array")  
    
        pygame.display.set_mode((500, 500))

 
        total_rewards = []
        for _ in range(num_episodes):
            state, info = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            while not (done or truncated):
                self.env.render()  
                action = np.argmax(self.q_table[state])
                state, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
            total_rewards.append(episode_reward)
        
        if render:
            self.env.close()
            self.env = gym.make(self.env.spec.id)
            self.env.reset()
        
        self.env.close()
        pygame.quit()

        return np.mean(total_rewards)
        
        


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False)
    np.set_printoptions(threshold=np.inf) 
    knowledge = rlang.parse_file("./frozen_lake.rlang")
    
    agent_with_policy = RLangDynaQAgent(env, n_planning_steps=50, knowledge=knowledge, policy_name="lake_policy", p_policy=0.1)
    rewards_with_policy = agent_with_policy.train(episodes=50)
    print(f"Average reward with policy: {agent_with_policy.test(10, render=False)}")
    agent = RLangDynaQAgent(env, n_planning_steps=50)
    rewards = agent.train(episodes=50)
    print(f"Average reward without policy: {agent.test(10, render=False)}")
    plot_training_rewards(rewards_with_policy,save_path="./plots/dyna_q_training_rewards_knowledge.png")
    plot_training_rewards(rewards,save_path="./plots/dyna_q_training_rewards.png")
    plot_comparison_training_rewards(
        reward_dict={
            "With RLang Policy": rewards_with_policy,
            "Without RLang": rewards
        },
        save_path="./plots/dyna_q_learning_comparison.png"
    )

