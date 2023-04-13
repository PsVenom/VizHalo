from stable_baselines3.common import env_checker
from Setup import HaloGym
import os
from stable_baselines3.common.callbacks import BaseCallback
CHECKPOINT_DIR = './train/train_basic'
LOG_DIR = './logs/log_basic'
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

env = HaloGym(render = True)
#training the model
model = PPO('MlpPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=2048)
model.learn(total_timesteps=600, callback=callback)
model = PPO.load('./train/train_basic/best_model_600')
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100)

