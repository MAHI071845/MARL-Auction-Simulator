from stable_baselines3 import PPO
from auction_env import AuctionEnv

env = AuctionEnv()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

model.save("auction_agent")
