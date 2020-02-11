#!/usr/bin/env python
# coding: utf-8

# In[43]:


import random
import _pickle as pickle
import numpy as np
import matplotlib.pyplot as plt


# In[44]:


from gym_tictactoe.env import TicTacToeEnv, set_log_level_by, agent_by_mark,    next_mark, check_game_status, after_action_state, O_REWARD, X_REWARD


# In[45]:


from utils import *

# In[46]:


class Agent:
    def __init__(self, no_actions=9, epsilon=0.01):
        """
        Q(s, a) = self.R[a][s]/self.num[a][s]
        """
        self.R = [{} for _ in range(9)]
        self.num = [{} for _ in range(9)]
        self.Policy = {}
        self.no_actions = no_actions
        self.epsilon = epsilon        
    
    def get_action(self, state, available_actions=list(range(9))):
        index = np.argmax([int(self.R[i].get(state) or 0)/int(self.num[i].get(state) or 1) for i in available_actions])
        return available_actions[index]
    
    def get_epsilon_greedy_action(self, state, available_actions=list(range(9))):
        index = np.argmax([int(self.R[i].get(state) or 0)/int(self.num[i].get(state) or 1) for i in available_actions])
        return available_actions[epsilon_greedy_action(len(available_actions), index, self.epsilon)]

    def get_optimal_prob(self, state, action, epsilon=0):
        if state not in self.R[action]: return 0
        weights = [int(self.R[i].get(state) or 0)/int(self.num[i].get(state) or 1) for i in range(len(self.R))]
        max_index = np.argmax(weights)
        return epsilon_greedy_prob(len(weights), max_index, action, epsilon)


# In[47]:


class Env:
    def __init__(self, player_get_action, opponent_get_action=None, env=TicTacToeEnv, player_first=True):
        self.env = env()
        self.env.set_start_mark('X')
        if opponent_get_action == None: self.opponent_get_action = random_action
        else: self.opponent_get_action = opponent_get_action
        self.player_get_action = player_get_action
        self.player_first = player_first
        
        if player_first:
            self.player_mark = 'X'
            self.opponent_mark = 'O'
        else:
            self.player_mark = 'O'
            self.opponent_mark = 'X'
        
    def state_to_str(self, state):
        return "".join(map(lambda x: str(x), state))  
        
    def player_play(self):
        init_state = self.state_to_str(self.env.board)
        action = self.player_get_action(init_state, self.env.available_actions())
        inter_state, reward, done, info = self.env.step(action)
        return (init_state, action ,reward)
    
    def opponent_play(self):
        init_state = (tuple(self.env.board), self.opponent_mark)
        action = self.opponent_get_action(init_state, self.env.available_actions())
        inter_state, reward, done, info = self.env.step(action)
        
    def generate_episode(self):
        self.env.reset()
        episode = []
        turn = True
        
        while not self.env.done:
            if self.player_first == turn: 
                init_state, action , reward = self.player_play()
                episode.append((init_state, action, int(self.env.done)))
            else:
                self.opponent_play()
                if self.env.done: episode[-1] = (episode[-1][0], episode[-1][1], -1)
            
            turn = not turn
        
        episode.append((self.state_to_str(self.env.board), None, None)) # Final state
        
        return episode
    
    def reset(self):
        self.env.reset()
        #self.env.set_start_mark('X')
        
    def play(self):
        if self.player_first: self.player_play()
        if not self.env.done: self.opponent_play()
        if not self.player_first: self.player_play()
        
        self.env.render()
        
        return self.env.done


# In[48]:


def on_policy_monte_carlo(agent, env_train, env_test, gamma=1.0, steps=10000, results_every=100, no_games=100):
    results = []
    unique_states = []
    for i in range(steps):
        
        if i % results_every == 0: 
            r = []
            for _ in range(no_games):  r.append(env_test.generate_episode()[-2][-1])
            results.append(r)
            states = [set(i.keys()) for i in agent.R]
            unique_states_ = set()
            for s in states:
                unique_states_ = unique_states_.union(s)
            unique_states.append(len(unique_states_))
            
        episode = env_train.generate_episode()
        reward = episode[-2][-1]
        d = 1
        
        for j, x in enumerate(zip(episode[:-1][::-1], episode[1:][::-1])):
            (s, a, r), (s1, _, _) = x
            agent.R[a][s] = int(agent.R[a].get(s) or 0) + reward * d
            agent.num[a][s] = int(agent.num[a].get(s) or 0) + 1
            d = d * gamma
            

    return unique_states, results


# In[49]:


def off_policy_monte_carlo(agent, env_train, env_test, behaviour_prob_func="random", gamma = 1.0, steps=10000, results_every=100, no_games=100):
    if not (behaviour_prob_func == "random"): raise NotImplementedError
    
    results = []
    unique_states = []
    
    c = [{} for _ in range(9)]
    
    for i in range(steps):
        
        if i % results_every == 0: 
            r = []
            for _ in range(no_games):  r.append(env_test.generate_episode()[-2][-1])
            results.append(r)
            states = [set(i.keys()) for i in agent.R]
            unique_states_ = set()
            for s in states:
                unique_states_ = unique_states_.union(s)
            unique_states.append(len(unique_states_))
            
        episode = env_train.generate_episode()
        reward = episode[-2][-1]
        g = 0
        w = 1
        d = 1
        
        for j, x in enumerate(episode[:-1][::-1]):
            (s, a, r) = x
            g = r * d
            c[a][s] = float(c[a].get(s) or 0) + w
            qas = int(agent.R[a].get(s) or 0)/int(agent.num[a].get(s) or 1)
            num = int(agent.num[a].get(s) or 1) + 1
            
            # qas = agent.R[a][s]/agent.num[a][s]
            agent.R[a][s] = (qas + w/float(c[a].get(s) or 1) * (g - qas)) * num
            
            # if behaviour_prob_func == "random"
            n = 2 * j + 9 - len(episode)
            optimal_action = np.argmax([agent.get_optimal_prob(s, a, 0) for a in range(9)])
            if optimal_action == a: break
            w = w * n # 1/n is random action's prob, n is the remaining no of actions
            
            d = d * gamma
            
            
            

    return unique_states, results

