from env.marl_env import AuctionMARL
from pettingzoo.utils.conversions import aec_to_parallel
import supersuit as ss
from stable_baselines3 import PPO

print("Creating MARL environment...")

env = AuctionMARL(n_agents=3)

# Convert AEC -> Parallel
env = aec_to_parallel(env)

# Convert to Gymnasium-compatible VecEnv
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

print("Training PPO agent...")

model = PPO("MlpPolicy", env, verbose=1, n_steps=64)
model.learn(total_timesteps=200000)

model.save("marl_agent")

print("Training finished!")
