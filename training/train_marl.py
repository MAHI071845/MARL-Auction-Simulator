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
import os
log_dir = "results/logs/"
os.makedirs(log_dir, exist_ok=True)


model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0001,
    n_steps=32,
    batch_size=32,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    tensorboard_log="results/logs/"
)

model.learn(total_timesteps=350000)

print("Model Saving Now")
model.save("marl_agent")
print("Training finished and model saved!")
