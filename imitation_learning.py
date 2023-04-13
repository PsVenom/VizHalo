from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from Setup import HaloGym
import os
from stable_baselines3.common.callbacks import BaseCallback
CHECKPOINT_DIR = './train/train_basic'
LOG_DIR = './logs/log_basic'
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from final_exec_torch import TrainAndLoggingCallback
import numpy as np
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

env = HaloGym(render = False)
rng = np.random.default_rng(0)

#training the model
model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=2048)
# model.learn(total_timesteps=600, callback=callback)
rollouts = rollout.rollout(
    model,
    DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
    rollout.make_sample_until(min_timesteps=None, min_episodes=50),
    rng=rng,
)
transitions = rollout.flatten_trajectories(rollouts)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)
bc_trainer.train(n_epochs=1)
reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
print("Reward:", reward)

