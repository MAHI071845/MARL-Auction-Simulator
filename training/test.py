from stable_baselines3 import PPO
from auction_env import AuctionEnv

env = AuctionEnv()
model = PPO.load("auction_agent")

obs, _ = env.reset()

done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    print("Action:", action, "Reward:", reward)
