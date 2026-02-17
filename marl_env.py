from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector
import numpy as np
import pandas as pd

class AuctionMARL(AECEnv):

    def __init__(self, num_agents=3):
        super().__init__()
        self.num_agents = num_agents
        self.agents = [f"team_{i}" for i in range(num_agents)]
        self.players = pd.read_csv("players.csv")

    def reset(self, seed=None, options=None):
        self.budgets = {a:100 for a in self.agents}
        self.index = 0
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.next()

    def step(self, action):
        agent = self.agent_selection
        player = self.players.iloc[self.index]

        if action == 1:
            cost = player.base_price
            if self.budgets[agent] >= cost:
                self.budgets[agent] -= cost

        self.index += 1
        self.agent_selection = self.agent_selector.next()
