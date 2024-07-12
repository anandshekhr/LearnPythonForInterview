import numpy as np
import gym
from gym import spaces

class JobShopEnv(gym.Env):
    def __init__(self, num_jobs, num_machines, processing_times):
        super(JobShopEnv, self).__init__()
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.processing_times = processing_times
        
        self.action_space = spaces.Discrete(num_jobs * num_machines)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(num_jobs, num_machines), dtype=np.float32)
        
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.zeros((self.num_jobs, self.num_machines))
        return self.state

    def step(self, action):
        job, machine = divmod(action, self.num_machines)
        done = False
        reward = -1  # Negative reward for each time step
        self.state[job, machine] += self.processing_times[job, machine]
        
        if np.all(self.state):
            done = True
            reward = 0  # Zero reward when all jobs are done
        
        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(self.state)


import random

class QLearningAgent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.num_actions))
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        self.q_table[state][action] += self.alpha * (td_target - self.q_table[state][action])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


num_jobs = 3
num_machines = 3
processing_times = np.array([
    [3, 2, 2],  # Job 1
    [2, 1, 4],  # Job 2
    [4, 3, 3]   # Job 3
])

env = JobShopEnv(num_jobs, num_machines, processing_times)
agent = QLearningAgent(num_states=num_jobs * num_machines, num_actions=num_jobs * num_machines)

num_episodes = 500

for episode in range(num_episodes):
    state = env.reset().flatten()
    total_reward = 0
    done = False

    while not done:
        state_idx = np.argmax(state)
        action = agent.choose_action(state_idx)
        next_state, reward, done, _ = env.step(action)
        next_state_idx = np.argmax(next_state.flatten())
        
        agent.update_q_table(state_idx, action, reward, next_state_idx)
        
        state = next_state.flatten()
        total_reward += reward

    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

print("Training completed.")


import matplotlib.pyplot as plt
import pandas as pd

def generate_gantt_chart(env, agent):
    state = env.reset().flatten()
    done = False
    schedule = []

    while not done:
        state_idx = np.argmax(state)
        action = agent.choose_action(state_idx)
        job, machine = divmod(action, env.num_machines)
        schedule.append((job, machine, env.processing_times[job, machine]))
        next_state, _, done, _ = env.step(action)
        state = next_state.flatten()

    df = pd.DataFrame(schedule, columns=['Job', 'Machine', 'Duration'])

    fig, ax = plt.subplots()
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for i, row in df.iterrows():
        ax.broken_barh([(i, row['Duration'])], (row['Machine']*10, 9), facecolors=colors[row['Job']])
    ax.set_ylim(0, env.num_machines*10)
    ax.set_xlim(0, sum(df['Duration']))
    ax.set_xlabel('Time')
    ax.set_yticks([15, 25, 35])
    ax.set_yticklabels(['Machine 1', 'Machine 2', 'Machine 3'])
    plt.show()

generate_gantt_chart(env, agent)
