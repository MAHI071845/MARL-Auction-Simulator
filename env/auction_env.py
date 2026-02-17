import gymnasium as gym
import numpy as np
import pandas as pd

class AuctionEnv(gym.Env):
    def __init__(self):
        self.players = pd.read_csv("data/players.csv")
        self.idx = 0
        self.budget = 100

        self.observation_space = gym.spaces.Box(0,200,(3,))
        self.action_space = gym.spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        self.idx = 0
        self.budget = 100
        return self._obs(), {}

    def _obs(self):
        p = self.players.iloc[self.idx]
        return np.array([p.rating,p.base_price,self.budget])

    def step(self, action):
        p = self.players.iloc[self.idx]
        cost = p.base_price + action*2

        if self.budget >= cost:
            self.budget -= cost
            reward = p.rating/cost
        else:
            reward = -5

        self.idx += 1
        done = self.idx >= len(self.players)

        obs = self._obs() if not done else np.zeros(3)
        return obs,reward,done,False,{}
