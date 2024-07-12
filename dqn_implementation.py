import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from dqn import JobShopEnv
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_dqn(env, num_episodes=1000, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, lr=0.001, batch_size=32):
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n
    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = deque(maxlen=2000)

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state.flatten(), dtype=torch.float32)
        total_reward = 0

        while True:
            if random.random() < epsilon:
                action = random.choice(range(action_dim))
            else:
                with torch.no_grad():
                    action = policy_net(state).argmax().item()

            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state.flatten(), dtype=torch.float32)
            memory.append((state, action, reward, next_state, done))

            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.stack(states)
                actions = torch.tensor(actions)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.stack(next_states)
                dones = torch.tensor(dones, dtype=torch.uint8)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(next_states).max(1)[0]
                target_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, target_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            total_reward += reward

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")

    return policy_net



# Define the job shop problem
num_jobs = 3
num_machines = 3

# Define processing times for each job on each machine
processing_times = np.array([
    [3, 2, 2],  # Job 1
    [2, 1, 4],  # Job 2
    [4, 3, 3]   # Job 3
])

import matplotlib.pyplot as plt
import pandas as pd

def generate_gantt_chart(env, policy_net):
    state = env.reset()
    state = torch.tensor(state.flatten(), dtype=torch.float32)
    done = False
    schedule = []

    while not done:
        with torch.no_grad():
            action = policy_net(state).argmax().item()
        job, machine = divmod(action, env.num_machines)
        schedule.append((job, machine, env.processing_times[job, machine]))
        next_state, _, done, _ = env.step(action)
        state = torch.tensor(next_state.flatten(), dtype=torch.float32)

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




env = JobShopEnv(num_jobs, num_machines, processing_times)
policy_net = train_dqn(env, num_episodes=500)
generate_gantt_chart(env, policy_net)

