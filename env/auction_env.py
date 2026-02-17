import gymnasium as gym
import numpy as np
import pandas as pd

class AuctionEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.players = pd.read_csv("players.csv")
        self.index = 0
        self.budget = 100

        # state = [rating, base_price, budget]
        self.observation_space = gym.spaces.Box(
            low=0, high=200, shape=(3,), dtype=np.float32
        )

        # 0 pass, 1 bid low, 2 bid medium, 3 bid high
        self.action_space = gym.spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        self.index = 0
        self.budget = 100
        return self._get_obs(), {}

    def _get_obs(self):
        p = self.players.iloc[self.index]
        return np.array([p.rating, p.base_price, self.budget], dtype=np.float32)

    def step(self, action):
        p = self.players.iloc[self.index]
        reward = 0

        if action == 0:
            reward -= 1
        else:
            cost = p.base_price + action * 2
            if self.budget >= cost:
                self.budget -= cost
                reward = p.rating / cost
            else:
                reward = -5

        self.index += 1
        done = self.index >= len(self.players)

        obs = self._get_obs() if not done else np.zeros(3)

        return obs, reward, done, False, {}
