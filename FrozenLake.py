#!/usr/bin/env python
# coding: utf-8

# In[91]:


import random
import numpy as np
import matplotlib.pyplot as plt


# In[92]:


import gym


# In[93]:


from utils import *

# In[177]:


class Agent:
    def __init__(self, no_actions=4, epsilon=0.01):
        """
        Q(s, a) = self.R[a][s]/self.num[a][s]
        """
        self.Q = np.array([[0. for _ in range(no_actions)] for _ in range(8 * 8)])
        #self.num = [{} for _ in range(9)]
        self.Policy = {}
        self.no_actions = no_actions
        self.epsilon = epsilon        
    
    def get_action(self, state, available_actions=list(range(4))):
        index = np.argmax(self.Q[state])
        return available_actions[index]
    
    def get_epsilon_greedy_action(self, state, available_actions=list(range(4)), epsilon=0.01):
        if epsilon > np.random.uniform(0, 1):
            return random.sample(available_actions, 1)[0]
        else:
            index = np.argmax(self.Q[state])
            return available_actions[index]


# In[180]:


def sarsa(agent, s, a, r, ns, na, alpha, gamma):
    agent.Q[s][a] =  (1 - alpha) * agent.Q[s][a] + alpha * (r + gamma * agent.Q[ns][na])


# In[181]:


def expected_sarsa(agent, s, a, r, ns, na, alpha, gamma):
    denominator = float(sum(agent.Q[ns])) + 10e-10
    probs = agent.Q[ns] / (denominator)
    agent.Q[s][a] =  (1 - alpha) * agent.Q[s][a] + alpha * (r + gamma * np.sum(probs * agent.Q[ns]))


# In[182]:


def q_learning(agent, s, a, r, ns, na, alpha, gamma):
    agent.Q[s][a] =  (1 - alpha) * agent.Q[s][a] + alpha * (r + gamma * np.max(agent.Q[ns]))


# In[222]:


def td_learning(agent, env, no_episodes=20000, gamma=0.5, 
                results_every=100, no_games=100, max_steps=1000, 
                update_func=sarsa, constant_alpha=False, alpha=0.5):
    
    if not constant_alpha: alpha = 1
    rewards = []
    count = 0
    for i in range(no_episodes):
        
        if i % results_every == 0:
            print(i, end='\r')
            total_r = []
            for j in range(no_games):
                r_ = 0
                env.reset()
                done, s = False, 0
                while not done:
                    a = agent.get_action(s)
                    s, r, done, _ = env.step(a)
                    r_ = r_ + r
                total_r.append(r_)
            rewards.append(total_r)
            
        env.reset()
        s, t = 0, 0
        a = agent.get_epsilon_greedy_action(s, epsilon=0.5)
        done = False
        while not done and t < max_steps:
            ns, r, done, _ = env.step(a)
            na = agent.get_epsilon_greedy_action(ns, epsilon=0.5)
            update_func(agent, s, a, r, ns, na, alpha, gamma)
            s = ns
            a = na
            t = t + 1
            if r > 0:
                count = count + 1
        if not constant_alpha: alpha = 1 / np.sqrt(i + 1)
                

            
    return rewards, count

