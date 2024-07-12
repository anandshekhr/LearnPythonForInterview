import gym
from gym import spaces
import numpy as np

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
