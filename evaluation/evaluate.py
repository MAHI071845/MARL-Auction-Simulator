import pandas as pd
import numpy as np
from stable_baselines3 import PPO

print("Loading trained model...")
model = PPO.load("marl_agent")

players = pd.read_csv("data/players.csv")

budget_rl = 100
budget_greedy = 100

team_rl = []
team_greedy = []

def greedy_decision(player, budget):
    if player.rating / player.base_price > 4 and budget >= player.base_price:
        return True
    return False

for _, player in players.iterrows():

    # -------- RL Agent --------
    obs = np.array([
        player.rating,
        player.base_price,
        budget_rl
    ], dtype=np.float32)

    action, _ = model.predict(obs)

    price_rl = player.base_price + int(action) * 2

    if budget_rl >= price_rl:
        budget_rl -= price_rl
        team_rl.append(player)

    # -------- Greedy Bot --------
    if greedy_decision(player, budget_greedy):
        budget_greedy -= player.base_price
        team_greedy.append(player)

# Compute team score
score_rl = sum(p.rating for p in team_rl)
score_greedy = sum(p.rating for p in team_greedy)

print("\n===== RESULTS =====")
print("RL Team Score:", score_rl)
print("Greedy Team Score:", score_greedy)
print("RL Budget Left:", budget_rl)
print("Greedy Budget Left:", budget_greedy)

if score_rl > score_greedy:
    print("🏆 RL Agent Wins!")
else:
    print("🏆 Greedy Bot Wins!")
