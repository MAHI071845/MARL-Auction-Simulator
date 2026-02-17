from stable_baselines3 import PPO
from env.auction_env import AuctionEnv

env = AuctionEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(20000)
model.save("single_agent")
