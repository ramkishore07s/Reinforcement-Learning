#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import copy
import _pickle as pickle
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


import gym


# In[ ]:


def preprocess_img(img):
    """
    Returns 80x80 image without the score at the top.
    """
    return np.mean(img, axis=2).astype(np.uint8)[::2, ::2][25:, :]


# In[ ]:


class RingBuffer:
    def __init__(self, size=100000):
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0
        
    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)
        
    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]
    
    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
            
    def sample(self, n=20):
        l = len(self)
        return [self[int(np.random.uniform(0, 1) * l)] for _ in range(n)]


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[ ]:


#torch.cuda.set_device(0)


# In[ ]:


class DQN(nn.Module):

    def __init__(self, h=80, w=80, outputs=4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        def outsize(size, kernel_size = 4, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        cw = outsize(outsize(outsize(w)))
        ch = outsize(outsize(outsize(h)))
        head_size = cw * ch * 32
        self.head = nn.Linear(head_size, 256)
        self.head2 = nn.Linear(256, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.head(x.view(x.size(0), -1)))
        return self.head2(x.view(x.size(0), -1))


# * With probability ε select a random action at
# ** otherwise select at = maxa Q∗(φ(st), a; θ)
# ** Execute action at in emulator and observe reward rt and image xt+1 Set st+1 = st, at, xt+1 and preprocess φt+1 = φ(st+1)
# ** Store transition (φt, at, rt, φt+1) in D
# ** Sample random minibatch of transitions (φj , aj , rj , φj +1 ) from D
# ** 􏰃 rj for terminal φj+1
# ** Set yj = rj + γ maxa′ Q(φj+1, a′; θ) for non-terminal φj+1

# In[ ]:


try: os.mkdir('models/')
except: pass


# In[ ]:


def train(dqn, target, env, max_epochs=10e5, replay_buf=RingBuffer(),
          test_every=1000, test_set=None, target_update_delay=10e3, gamma=1.0,
          save_every=10e3, recent=False, var_epsilon=False):
    
    epoch = 0
    episode = 0
    if not var_epsilon: epsilon = 0.1
    else: epsilon, diff = 1.0, (1.0 - 0.1)/10e5
    scores = []
    huber = nn.SmoothL1Loss()
    optimizer = optim.RMSprop(dqn.parameters())
    total_reward = 0

    #for episode in range(no_episodes):
    while epoch < max_epochs:
        lives = 5
        episode_score = 0
        episode = episode + 1
        input_buf = []
        frame = env.reset()
        is_done = False
        for _ in range(4): input_buf.append(preprocess_img(frame))
            
        while not is_done:
            if epoch % target_update_delay == 0:
                target.load_state_dict(dqn.state_dict())
                
            optimizer.zero_grad()
            dqn.zero_grad()

            if np.random.uniform(0, 1) < epsilon: 
                action = env.action_space.sample()
            else:
                input = torch.cuda.FloatTensor(np.array([input_buf])/256)
                action = torch.argmax(dqn(input)).cpu().numpy()

            next_input_buf = []
            reward = 0
            for _ in range(4): 
                frame, r, is_done, life = env.step(action)
                next_input_buf.append(preprocess_img(frame))
                if lives > life['ale.lives']: r, lives = -1, life['ale.lives']
                reward += r
                episode_score += r
                total_reward += r
                

            replay_buf.append([np.array(input_buf), action, reward, is_done, np.array(next_input_buf)])
            
            if recent:
                sampled_replay = replay_buf.sample(31)
                states = [sampled_replay[i][0] for i in range(31)]
                states.append(input_buf)
                actions = [[sampled_replay[i][1]] for i in range(31)]
                actions.append([action])
                rewards = [sampled_replay[i][2] for i in range(31)]
                rewards.append(reward)
                is_terminal = [sampled_replay[i][3] for i in range(31)]
                is_terminal.append(is_done)
                next_states = [sampled_replay[i][4] for i in range(31)]
                next_states.append(input_buf)
            else:
                sampled_replay = replay_buf.sample(32)
                states = [sampled_replay[i][0] for i in range(32)]
                actions = [[sampled_replay[i][1]] for i in range(32)]
                rewards = [sampled_replay[i][2] for i in range(32)]
                is_terminal = [sampled_replay[i][3] for i in range(32)]
                next_states = [sampled_replay[i][4] for i in range(32)]

            next_states_mask = torch.cuda.FloatTensor(is_terminal)
            
            with torch.no_grad():
                next_state_qs = torch.max(target(torch.cuda.FloatTensor(next_states)), dim=-1)[0] * next_states_mask

            output_mask = torch.zeros(32, 4).cuda()
            for i, action in enumerate(actions): output_mask[i][action] = 1

            outputs = dqn(torch.cuda.FloatTensor(np.array(states)/256))
            predicted_qs = torch.sum(output_mask * outputs, dim=1)
            actual_qs = torch.cuda.FloatTensor(rewards) + gamma * next_state_qs

            loss = huber(predicted_qs, actual_qs)
            loss.backward()
            print(episode, lives, str(total_reward/episode)[:7], episode_score, epoch, action[0], loss.detach().cpu().numpy(), end='\r')

            optimizer.step()

            input_buf = next_input_buf
            epoch = epoch + 1
            if var_epsilon: epsilon = epsilon - diff
        
            if epoch % test_every == 1:
                if test_set is not None:
                    with torch.no_grad():
                        scores.append(np.mean(dqn(torch.cuda.FloatTensor(test_set)).cpu().numpy()))

            if epoch % save_every == 0:
                print('')
                print("saving model: ", epoch)
                #torch.save(dqn.state_dict(), "models/dqn_r_" + str(epoch) + ".pkl")
                #pickle.dump(scores, open('scores_r.pkl', 'wb+'))
                #pickle.dump(replay_buf, open('episode_replay_r.pkl', 'wb+'))

    return scores


# In[ ]:


def train_q2(dqn, target, env, max_epochs=10e5, replay_buf=RingBuffer(),
          test_every=1000, test_set=None, target_update_delay=10e3, gamma=0.7,
          save_every=10e3, recent=False, var_epsilon=True):
    
    epoch = 0
    episode = 0
    if not var_epsilon: epsilon = 0.1
    else: epsilon, diff = 1.0, (1.0 - 0.1)/10e5
    scores = []
    huber = nn.SmoothL1Loss()
    optimizer_dqn = optim.RMSprop(dqn.parameters())
    optimizer_target = optim.RMSprop(target.parameters())
    total_reward = 0

    #for episode in range(no_episodes):
    while epoch < max_epochs:
        lives = 5
        episode_score = 0
        episode = episode + 1
        input_buf = []
        frame = env.reset()
        is_done = False
        for _ in range(4): input_buf.append(preprocess_img(frame))
            
        while not is_done:
            if np.random.uniform(0, 1) >= 0.5: m1 = True
            else: m1 = False

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            dqn.zero_grad()
            target.zero_grad()

            if np.random.uniform(0, 1) < epsilon: 
                action = env.action_space.sample()
            else:
                input = torch.cuda.FloatTensor(np.array([input_buf])/256)
                if m1: action = torch.argmax(dqn(input)).cpu().numpy()
                else: action = torch.argmax(target(input)).cpu().numpy()

            next_input_buf = []
            reward = 0
            for _ in range(4): 
                frame, r, is_done, life = env.step(action)
                next_input_buf.append(preprocess_img(frame))
                if lives > life['ale.lives']: r, lives = -1, life['ale.lives']
                reward += r
                episode_score += r
                total_reward += r
                

            replay_buf.append([np.array(input_buf), action, reward, is_done, np.array(next_input_buf)])
            
            if recent:
                sampled_replay = replay_buf.sample(31)
                states = [sampled_replay[i][0] for i in range(31)]
                states.append(input_buf)
                actions = [[sampled_replay[i][1]] for i in range(31)]
                actions.append([action])
                rewards = [sampled_replay[i][2] for i in range(31)]
                rewards.append(reward)
                is_terminal = [sampled_replay[i][3] for i in range(31)]
                is_terminal.append(is_done)
                next_states = [sampled_replay[i][4] for i in range(31)]
                next_states.append(input_buf)
            else:
                sampled_replay = replay_buf.sample(32)
                states = [sampled_replay[i][0] for i in range(32)]
                actions = [[sampled_replay[i][1]] for i in range(32)]
                rewards = [sampled_replay[i][2] for i in range(32)]
                is_terminal = [sampled_replay[i][3] for i in range(32)]
                next_states = [sampled_replay[i][4] for i in range(32)]

            next_states_mask = torch.cuda.FloatTensor(is_terminal)

            output_mask = torch.zeros(32, 4).cuda()
            for i, action in enumerate(actions): output_mask[i][action] = 1
            
            with torch.no_grad():
                if m1: next_state_qs = torch.max(target(torch.cuda.FloatTensor(next_states)), dim=-1)[0] * next_states_mask
                else: next_state_qs = torch.max(dqn(torch.cuda.FloatTensor(next_states)), dim=-1)[0] * next_states_mask
                    

            if m1:
                outputs = dqn(torch.cuda.FloatTensor(np.array(states)/256))
            else: 
                outputs = target(torch.cuda.FloatTensor(np.array(states)/256))
                
            predicted_qs = torch.sum(output_mask * outputs, dim=1)
            actual_qs = torch.cuda.FloatTensor(rewards) + gamma * next_state_qs

            loss = huber(predicted_qs, actual_qs)
            loss.backward()
            

            if m1: optimizer1.step()
            else: optimizer2.step()

            print(episode, epsilon, lives, str(total_reward/episode)[:7], episode_score, epoch, action[0], loss.detach().cpu().numpy(), end='\r')
            input_buf = next_input_buf
            epoch = epoch + 1
            if var_epsilon: epsilon = epsilon - diff
        
            if epoch % test_every == 1:
                if test_set is not None:
                    with torch.no_grad():
                        scores.append(np.mean(dqn(torch.cuda.FloatTensor(test_set)).cpu().numpy()))

            if epoch % save_every == 0:
                print('')
                print("saving model: ", epoch)
                torch.save(dqn.state_dict(), "models/dqn1_2q_" + str(epoch) + ".pkl")
                torch.save(target.state_dict(), "models/dqn2_2q_" + str(epoch) + ".pkl")
                pickle.dump(scores, open('scores_2q.pkl', 'wb+'))
                pickle.dump(replay_buf, open('episode_replay_2q.pkl', 'wb+'))

    return scores


# ```
# dqn = DQN()
# target = DQN()
# target.cuda()
# dqn.cuda()
# ```

# In[ ]:


def get_test_set():
    env = gym.make('BreakoutDeterministic-v4')

    test_set = []
    frame = env.reset()

    input_buf = []
    for _ in range(4): input_buf.append(preprocess_img(frame))
    test_set.append(input_buf)

    is_done = False
    for _ in range(100):
        action = env.action_space.sample()
        input_buf = []
        for _ in range(4):
            frame, reward, is_done, _ = env.step(action)
            input_buf.append(preprocess_img(frame))
        test_set.append(input_buf)
    return test_set


# In[ ]:


#scores = train(dqn, target, env, test_set=test_set, replay_buf=r, recent=True)


# In[ ]:


def test_model(dqn, save_as='project.mp4'):
    env = gym.make('BreakoutDeterministic-v4')

    frames = []
    score = 0
    i = 0
    input_buf = []

    with torch.no_grad():
        frame = env.reset()
        frames.append(frame)
        is_done = False
        for _ in range(4): input_buf.append(preprocess_img(frame))

        while not is_done:
            next_input_buf = []
            i = i + 1
            if np.random.uniform(0, 1) < 0.05:
                pred = dqn(torch.cuda.FloatTensor([input_buf])).cpu().numpy()
                action = np.argmax(pred)
            else:
                action = env.action_space.sample()
            for _ in range(4):
                frame, reward, is_done, _ = env.step(action)
                frames.append(frame)
                score = score + reward
                next_input_buf.append(preprocess_img(frame))

            input_buf = next_input_buf
            
    import cv2

    img_array = []
    for img in frames:
        img = img
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img) 

    out = cv2.VideoWriter(save_as ,cv2.VideoWriter_fourcc(*'MP4V'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    return frames, score


# In[ ]:




