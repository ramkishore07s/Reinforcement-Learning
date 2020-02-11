#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import random


# In[24]:


def random_action(_, actions):
    """
    Probs: array with probabilities
    Output: Index of the action selected
    """
    random.shuffle(actions)
    return actions[random.sample(list(range(len(actions))), 1)[0]]


# In[31]:


def epsilon_greedy_action(length, index, epsilon=0.1):
    bin_size = epsilon/length
    random_no = np.random.rand()
    
    if random_no < (index) * bin_size:
        return int(random_no/bin_size)
    elif random_no <= (index + 1)* bin_size + 1 - epsilon:
        return index
    else:
        return int((random_no - 1 + epsilon)/bin_size)


# In[ ]:


def epsilon_greedy_prob(length, max_index, action_index, epsilon=0.1):
    if max_index == action_index: return 1 - epsilon + epsilon/length
    return epsilon/length


# In[ ]:


def plot_mean_and_CI(rewards, color_mean=None, color_shading=None):
    mean = np.mean(rewards, axis=-1)
    lb = -np.abs(np.std(rewards, axis=-1)) + mean
    ub = np.abs(np.std(rewards, axis=-1)) + mean
    # plot the shaded range of the confidence intervals
    if color_shading is not None: plt.fill_between(mean, ub, -mean, color=color_shading, alpha=.8)
    # plot the mean on top
    plt.plot(mean, color_mean)


# In[ ]:


def plot_table(results, colLabels=['On Policy', 'Off Policy']):
    fig, axs = plt.subplots()
    axs.axis('tight')
    axs.axis('off')
    table = axs.table(cellText=[[np.mean(r) for r in results],
                        [np.std(r) for r in results]], loc='center',rowLabels=['mean', 'std'],
                          colLabels=colLabels)
    fig.tight_layout()
    plt.plot()

