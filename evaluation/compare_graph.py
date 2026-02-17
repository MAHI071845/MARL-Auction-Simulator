import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

model = PPO.load("marl_agent")
players = pd.read_csv("data/players.csv")

rl_scores = []
greedy_scores = []

for episode in range(20):

    budget_rl = 100
    budget_greedy = 100

    team_rl = []
    team_greedy = []

    for _, p in players.sample(frac=1).iterrows():

        # RL
        obs = np.array([p.rating, p.base_price, budget_rl], dtype=np.float32)
        action, _ = model.predict(obs)
        price = p.base_price + int(action)*2
        if budget_rl >= price:
            budget_rl -= price
            team_rl.append(p.rating)

        # Greedy
        if p.rating/p.base_price > 4 and budget_greedy >= p.base_price:
            budget_greedy -= p.base_price
            team_greedy.append(p.rating)

    rl_scores.append(sum(team_rl))
    greedy_scores.append(sum(team_greedy))

plt.plot(rl_scores, label="RL Agent")
plt.plot(greedy_scores, label="Greedy Bot")
plt.legend()
plt.title("RL vs Greedy Team Score")
plt.xlabel("Episode")
plt.ylabel("Team Score")
plt.show()
