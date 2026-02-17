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

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # ---------------- RESET ----------------
    def reset(self, seed=None, options=None):

        self.agents = self.possible_agents[:]
        self.budgets = {a: 100 for a in self.agents}
        self.team_value = {a: 0 for a in self.agents}

        self.rewards = {a: 0 for a in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self.idx = 0
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    # ---------------- OBSERVE ----------------
    def observe(self, agent):

        if self.idx >= len(self.players):
            return np.zeros(3, dtype=np.float32)

        player = self.players.iloc[self.idx]

        return np.array(
            [player.rating, player.base_price, self.budgets[agent]],
            dtype=np.float32,
        )

    # ---------------- STEP ----------------
    def step(self, action):

        if self.idx >= len(self.players):
            return

        agent = self.agent_selection
        player = self.players.iloc[self.idx]

        cost = player.base_price + action * 2

        # -------- Improved Reward --------
        if self.budgets[agent] >= cost:

            self.budgets[agent] -= cost
            self.team_value[agent] += player.rating

            # Value-for-money reward
            value_score = player.rating / (cost + 1)

            # Budget efficiency reward
            budget_penalty = 0.01 * (100 - self.budgets[agent])

            reward = value_score * 10 - budget_penalty

        else:
            reward = -5  # overspending penalty

        self.rewards[agent] = reward
        self._cumulative_rewards[agent] += reward

        # Move to next agent
        if self._agent_selector.is_last():
            self.idx += 1

            # -------- End Auction Bonus --------
            if self.idx >= len(self.players):

                for a in self.agents:
                    team_bonus = self.team_value[a] * 0.05
                    self._cumulative_rewards[a] += team_bonus

                # Restart auction for continuous training
                self.idx = 0
                self.budgets = {a: 100 for a in self.agents}
                self.team_value = {a: 0 for a in self.agents}

        self.agent_selection = self._agent_selector.next()
