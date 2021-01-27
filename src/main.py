#!/usr/bin/env python3
import cira
import gym 
import numpy as np
import matplotlib
matplotlib.use('Agg') # no UI backend

import matplotlib.pyplot as plt

"""
Using Q learning to trade stocks
Actions: buy, sell, hold  
"""

__author__ = "Axel Gard"
__version__ = "0.1.0"
__license__ = "MIT"


cira.alpaca.KEY_FILE = "../test_key.json"

portfolio = cira.Portfolio()
exchange = cira.Exchange()

env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1 # 0-1
DISCOUNT = 0.95 # how important is futre actions 0-1
EPISODES = 2000

SHOW_EVERY = 500

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

epsilon = 0.5 # 0-1 change of ramdom action 
START_EPSILON_DECAYING = 1 
END_EPSILON_DECAYING  = EPISODES // 2 

epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])) # whant to change value 

ep_rewards = []
aggr_ep_rewards = {'ep':[], 'avg':[], 'min':[], 'max':[]}

def get_discrite_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    episode_reward = 0
    if episode % SHOW_EVERY == 0: 
        print(episode)
        render = True
    else:
        render = False

    discrete_state = get_discrite_state(env.reset())
    done = False
    while not done: 
        if np.random.random() > epsilon: 
            action = np.argmax(q_table[discrete_state]) 
        else: 
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrite_state(new_state)
        if render:
            env.render()

        if not done:
            max_futrue_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_futrue_q) # Q formula 
            q_table[discrete_state + (action, )] = new_q

        elif new_state[0] >= env.goal_position:
            print(f"we made int on {episode}") 
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value 

    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY: # == 0 
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
fig = plt.figure()
fig.savefig('plot.png')