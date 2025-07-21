import sys

sys.path.append('../../DDPG-CARLA')
sys.path.append('../../DDPG-CARLA/DDPG')

import argparse
import time
import keras.backend as keras_backend
import numpy as np
import tensorflow as tf
from DDPG.carla_env import CarEnv
from DDPG.actor import ActorNetwork
from DDPG.critic import CriticNetwork
from DDPG.actor_CNN import ActorNetwork_CNN
from DDPG.critic_CNN import CriticNetwork_CNN

from keras.callbacks import TensorBoard
# from util.noise import OrnsteinUhlenbeckActionNoise
from DDPG.replay_buffer import ReplayBuffer
import cv2
import carla_config as settings
from keras.models import load_model

time_buff = []

AGGREGATE_STATS_EVERY = 10


class ModifiedTensorBoard(TensorBoard):

  # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.step = 1
    self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)
    # self.writer = tf.summary.FileWriter(self.log_dir)

  # Overriding this method to stop creating default log writer
  def set_model(self, model):
    pass

  # Overrided, saves logs with our step number
  # (otherwise every .fit() will start writing from 0th step)
  def on_epoch_end(self, epoch, logs=None):
    self.update_stats(**logs)

  # Overrided
  # We train for one batch only, no need to save anything at epoch end
  def on_batch_end(self, batch, logs=None):
    pass

  # Overrided, so won't close writer
  def on_train_end(self, _):
    pass

  # Custom method for saving own metrics
  # Creates writer, writes custom metrics and closes writer
  def update_stats(self, **stats):
    self._write_logs(stats, self.step)


def preprocess():
  global tensorboard, step, buffer, env, ep_rewards

  tensorboard = ModifiedTensorBoard(
    log_dir=f"logs/logs_{settings.WORKING_MODE}/{settings.TRAIN_MODE}-{int(time.time())}")
  step = 0

  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  tf_session = tf.compat.v1.Session(config=config)

  tf.compat.v1.keras.backend.set_session(tf_session)

  if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == \
    settings.WORKING_MODE_OPTIONS[1] \
    or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7] or settings.WORKING_MODE == \
    settings.WORKING_MODE_OPTIONS[8]:
    actor = ActorNetwork(tf_session=tf_session, state_size=settings.state_dim, action_size=3,
                         tau=settings.tau, lr=settings.lra)
    critic = CriticNetwork(tf_session=tf_session, state_size=settings.state_dim, action_size=3,
                           tau=settings.tau, lr=settings.lrc)
  else:
    actor = ActorNetwork_CNN(tf_session=tf_session, tau=settings.tau, lr=settings.lra)
    critic = CriticNetwork_CNN(tf_session=tf_session, tau=settings.tau, lr=settings.lrc)

  buffer = ReplayBuffer(settings.buffer_size)

  env = CarEnv()

  try:
    # print(settings.critic_weights_file)
    actor.model.load_weights(settings.actor_weights_file)
    critic.model.load_weights(settings.critic_weights_file)
    actor.target_model.load_weights(settings.actor_weights_file)
    critic.target_model.load_weights(settings.critic_weights_file)
    print("Weights loaded successfully")
  except:
    print("Cannot load weights")

  ep_rewards = []
