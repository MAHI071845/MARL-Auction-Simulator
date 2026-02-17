from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
import gymnasium as gym
import numpy as np
import pandas as pd


class AuctionMARL(AECEnv):

    metadata = {
        "name": "auction_marl",
        "is_parallelizable": True,
        "render_modes": ["human"],
    }

    render_mode = None

    def __init__(self, n_agents=3):
        super().__init__()

        self.players = pd.read_csv("data/players.csv")
        self.possible_agents = [f"team_{i}" for i in range(n_agents)]

        # Observation: [rating, base_price, budget]
        self.observation_spaces = {
            agent: gym.spaces.Box(low=0, high=200, shape=(3,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # Actions: pass / small / medium / high bid
        self.action_spaces = {
            agent: gym.spaces.Discrete(4)
            for agent in self.possible_agents
        }

    # Required functions
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # Reset environment
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.budgets = {a: 100 for a in self.agents}

        self.rewards = {a: 0 for a in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self.idx = 0
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    # Safe observation
    def observe(self, agent):
        if self.idx >= len(self.players):
            return np.zeros(3, dtype=np.float32)

        player = self.players.iloc[self.idx]
        return np.array(
            [player.rating, player.base_price, self.budgets[agent]],
            dtype=np.float32,
        )

    # Step function
    def step(self, action):

        # If auction finished
        if self.idx >= len(self.players):
            return

        agent = self.agent_selection
        player = self.players.iloc[self.idx]

        cost = player.base_price + action * 2

        if self.budgets[agent] >= cost:
            self.budgets[agent] -= cost
            reward = player.rating / cost
        else:
            reward = -5

        self.rewards[agent] = reward
        self._cumulative_rewards[agent] += reward

        # Move to next agent
        if self._agent_selector.is_last():
            self.idx += 1

            # End auction after all agents finished last player
            if self.idx >= len(self.players):
                for a in self.agents:
                    self.terminations[a] = True

        self.agent_selection = self._agent_selector.next()
