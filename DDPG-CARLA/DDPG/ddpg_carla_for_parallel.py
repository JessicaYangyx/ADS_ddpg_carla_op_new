# import argparse
# import sys
# import time
# import keras.backend as keras_backend
# import numpy as np
# import tensorflow as tf
# from DDPG.carla_env import CarEnv
# from DDPG.actor import ActorNetwork
# from DDPG.critic import CriticNetwork
# from DDPG.actor_CNN import ActorNetwork_CNN
# from DDPG.critic_CNN import CriticNetwork_CNN
# from tensorflow.keras.callbacks import TensorBoard
# from DDPG.replay_buffer import ReplayBuffer
# import cv2
# import carla_config as settings
# from keras.models import load_model
# from typing import Any
# import random
# import carla
# from multiprocessing import Process, Queue, Pipe, Array, Value
#
# time_buff = []
# AGGREGATE_STATS_EVERY = 10
#
#
# class ModifiedTensorBoard(TensorBoard):
#   def __init__(self, **kwargs):
#     super().__init__(**kwargs)
#     self.step = 1
#     self.writer = tf.summary.create_file_writer(self.log_dir)
#     self._should_write_train = True
#
#   def set_model(self, model):
#     pass
#
#   def on_epoch_end(self, epoch, logs=None):
#     self.update_stats(**logs) if logs else None
#
#   def on_batch_end(self, batch, logs=None):
#     pass
#
#   def on_train_end(self, _):
#     pass
#
#   def update_stats(self, **stats):
#     with self.writer.as_default():
#       for name, value in stats.items():
#         tf.summary.scalar(name, value, step=self.step)
#
#
# def play(train_indicator, q: Queue):
#   tf.keras.backend.clear_session()
#   tf.config.experimental_run_functions_eagerly(True)
#   physical_devices = tf.config.list_physical_devices('GPU')
#   [tf.config.experimental.set_memory_growth(dev, True) for dev in physical_devices]
#
#   tensorboard = ModifiedTensorBoard(
#     log_dir=f"logs/logs_{settings.WORKING_MODE}/{settings.TRAIN_MODE}-{int(time.time())}")
#   step = 0
#
#   config = tf.compat.v1.ConfigProto()
#   config.gpu_options.allow_growth = True
#   tf_session = tf.compat.v1.Session(config=config)
#   tf.compat.v1.keras.backend.set_session(tf_session)
#
#   if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == \
#     settings.WORKING_MODE_OPTIONS[1] \
#     or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7] or settings.WORKING_MODE == \
#     settings.WORKING_MODE_OPTIONS[8]:
#     actor = ActorNetwork(tf_session=tf_session, state_size=settings.state_dim, action_size=3,
#                          tau=settings.tau, lr=settings.lra)
#     critic = CriticNetwork(tf_session=tf_session, state_size=settings.state_dim, action_size=3,
#                            tau=settings.tau, lr=settings.lrc)
#   else:
#     actor = ActorNetwork_CNN(tf_session=tf_session, tau=settings.tau, lr=settings.lra)
#     critic = CriticNetwork_CNN(tf_session=tf_session, tau=settings.tau, lr=settings.lrc)
#
#   buffer = ReplayBuffer(settings.buffer_size)
#   env = CarEnv()
#
#   try:
#     actor.model.load_weights(settings.actor_weights_file)
#     critic.model.load_weights(settings.critic_weights_file)
#     actor.target_model.load_weights(settings.actor_weights_file)
#     critic.target_model.load_weights(settings.critic_weights_file)
#     print("Weights loaded successfully")
#   except:
#     print("Cannot load weights")
#
#   ep_rewards = []
#
#   for i in range(settings.episodes_num):
#     tensorboard.step = i
#     print("Episode : %s Replay buffer %s" % (i, len(buffer)))
#
#     # 从bridge进程获取车道线数据
#     lane_data = None
#     if not q.empty():
#       lane_data = q.get()
#       print("Received lane data from bridge:", lane_data)
#       # 在这里使用车道线数据进行训练决策
#
#     if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == \
#       settings.WORKING_MODE_OPTIONS[1] \
#       or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8]:
#       _, state = env.reset()
#     else:
#       current_state, _ = env.reset()
#       if settings.IM_LAYERS == 1:
#         current_state = np.expand_dims(current_state, -1)
#       current_state = current_state / 255
#       if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7]:
#         current_state = current_state.flatten()
#
#     if settings.GUARDAR_DATOS == 1:
#       np.savetxt(
#         'Waypoints/DDPG_wp_' + str(settings.WORKING_MODE) + '_' + str(settings.TRAIN_MODE) + '_' + str(i) + '.txt',
#         env.waypoints_txt, delimiter=';')
#
#     total_reward = 0.0
#     for j in range(settings.max_steps):
#       tm1 = time.time()
#       loss = 0
#
#       # 获取最新的车道线数据
#       if not q.empty():
#         lane_data = q.get_nowait()
#         # print("Updated lane data:", lane_data)
#
#       if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == \
#         settings.WORKING_MODE_OPTIONS[1] \
#         or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8]:
#         action_predicted = actor.model.predict(state.reshape(1, state.shape[0]))
#         [new_image, new_state], reward, done, info = env.step(action_predicted[0])
#         buffer.add((state, action_predicted[0], reward, new_state, done))
#       else:
#         action_predicted = actor.model.predict(np.array(current_state).reshape(-1, *current_state.shape))
#         [new_current_state, _], reward, done, info = env.step(action_predicted[0])
#         if settings.IM_LAYERS == 1:
#           new_current_state = np.expand_dims(new_current_state, -1)
#         new_current_state = new_current_state / 255
#         if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7]:
#           new_current_state = new_current_state.flatten()
#         buffer.add((current_state, action_predicted[0], reward, new_current_state, done))
#
#       batch = buffer.get_batch(settings.batch_size)
#       states = np.asarray([e[0] for e in batch])
#       actions = np.asarray([e[1] for e in batch])
#       rewards = np.asarray([e[2] for e in batch])
#       new_states = np.asarray([e[3] for e in batch])
#       dones = np.asarray([e[4] for e in batch])
#       y_t = np.zeros((len(batch), 1))
#
#       target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])
#
#       for k in range(len(batch)):
#         if dones[k]:
#           y_t[k] = rewards[k]
#         else:
#           y_t[k] = rewards[k] + settings.gamma * target_q_values[k]
#
#       if train_indicator:
#         loss += critic.model.train_on_batch([states, actions], y_t)
#         a_for_grad = actor.model.predict(states)
#         grads = critic.get_gradients(states, a_for_grad)
#         actor.train(states, grads)
#         actor.train_target_model()
#         critic.train_target_model()
#
#       total_reward += reward
#
#       if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == \
#         settings.WORKING_MODE_OPTIONS[1] \
#         or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8]:
#         state = new_state
#       else:
#         current_state = new_current_state
#
#       print("Episode %s - Step %s - Action %s - Reward %s" % (i, step, action_predicted[0], reward))
#       step += 1
#       if done:
#         print(env.summary)
#         break
#
#     ep_rewards.append(total_reward)
#     if (i > 0) and ((i % AGGREGATE_STATS_EVERY == 0) or (i == 1)):
#       average_reward = np.mean(ep_rewards[-AGGREGATE_STATS_EVERY:])
#       min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
#       max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
#       average_dist = np.mean(env.distance_acum[-AGGREGATE_STATS_EVERY:])
#       tensorboard.update_stats(average_reward=average_reward, min_reward=min_reward, max_reward=max_reward,
#                                distance=average_dist, loss=loss)
#
#     if i % 3 == 0 and train_indicator:
#       actor.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_actor.h5", overwrite=True)
#       critic.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_critic.h5", overwrite=True)
#     if i % settings.N_save_stats == 0 and train_indicator:
#       actor.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_" + str(i) + "_actor.h5",
#                                overwrite=True)
#       critic.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_" + str(i) + "_critic.h5",
#                                 overwrite=True)
#     if (i > 10) and (total_reward > np.max(ep_rewards[:-1])):
#       actor.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_best_reward_actor.h5",
#                                overwrite=True)
#       critic.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_best_reward_critic.h5",
#                                 overwrite=True)
#
#     if settings.GUARDAR_DATOS == 1:
#       np.savetxt(
#         'Trayectorias/DDPG_trayectoria_' + str(settings.WORKING_MODE) + '_' + str(settings.TRAIN_MODE) + '_' + str(
#           i) + '.txt',
#         env.position_array, delimiter=';')
#       env.position_array = []
#
#     time_buff.append((time.time() - tm1))
#     tm = time.strftime("%Y-%m-%d %H:%M:%S")
#     episode_stat = "%s -th Episode. %s total steps. Total reward: %s. Time %s" % (i, step, total_reward, tm)
#     dif_time = "Step time %s" % (time.time() - tm1)
#     print(episode_stat)
#
#     for actor_world in env.actor_list:
#       actor_world.destroy()
#
#   actor.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_actor.h5", overwrite=True)
#   critic.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_critic.h5", overwrite=True)
#
#
# if __name__ == "__main__":
#   parser = argparse.ArgumentParser()
#   parser.add_argument("-t", "--train", type=int, help="train indicator", default=1)
#   args = parser.parse_args()
#
#   # generate vehicle
#   client = carla.Client("localhost", 2000)
#   world = client.get_world()
#   blueprint_library = world.get_blueprint_library()
#   model_3 = blueprint_library.filter("model3")[0]
#   model_3.set_attribute('role_name', 'init')
#   transform = random.choice(world.get_map().get_spawn_points())
#   vehicle = world.spawn_actor(model_3, transform)
#
#   # 创建共享队列
#   q = Queue()
#
#   # 启动bridge进程
#   sys.path.append("/home/yanxi/carla/openpilot_0.8.9/openpilot0.8.9")
#   sys.path.append("/home/yanxi/carla/openpilot_0.8.9/openpilot0.8.9/tools/sim")
#   from bridge_for_parallel import bridge
#
#   Radar_Point_Array = Array('d', np.array([0] * (64 * 4 + 16 * 3 + 1)).astype('float64'))
#   N_radar_points = Value('i', 0)
#
#   bridge_proc = Process(target=bridge, args=(q, Radar_Point_Array, N_radar_points), daemon=True)
#   bridge_proc.start()
#
#   # 启动主训练进程
#   play_proc = Process(target=play, args=(args.train, q), daemon=True)
#   play_proc.start()
#
#   try:
#     while True:
#       time.sleep(1)
#   except KeyboardInterrupt:
#     print("Shutting down...")
#     bridge_proc.terminate()
#     play_proc.terminate()
#     bridge_proc.join()
#     play_proc.join()


import argparse
import sys
import time
import keras.backend as keras_backend
import numpy as np
import tensorflow as tf
from DDPG.carla_env import CarEnv
from DDPG.actor import ActorNetwork
from DDPG.critic import CriticNetwork
from DDPG.actor_CNN import ActorNetwork_CNN
from DDPG.critic_CNN import CriticNetwork_CNN
from tensorflow.keras.callbacks import TensorBoard
from DDPG.replay_buffer import ReplayBuffer
import cv2
import carla_config as settings
from keras.models import load_model
from typing import Any
import random
import carla
from multiprocessing import Process, Queue, Pipe, Array, Value

AGGREGATE_STATS_EVERY = 10
time_buff = []


class DDPGTrainer:
  def __init__(self, q):
    self.q = q
    self.setup_tensorflow()
    self.initialize_networks()
    self.env = CarEnv()
    self.buffer = ReplayBuffer(settings.buffer_size)
    self.step = 0

  def setup_tensorflow(self):
    tf.keras.backend.clear_session()
    tf.config.experimental_run_functions_eagerly(True)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    self.tf_session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(self.tf_session)

  def initialize_networks(self):
    print("Initializing networks...")
    if settings.WORKING_MODE in [settings.WORKING_MODE_OPTIONS[0], settings.WORKING_MODE_OPTIONS[1],
                                 settings.WORKING_MODE_OPTIONS[7], settings.WORKING_MODE_OPTIONS[8]]:
      self.actor = ActorNetwork(
        tf_session=self.tf_session,
        state_size=settings.state_dim,
        action_size=3,
        tau=settings.tau,
        lr=settings.lra
      )
      self.critic = CriticNetwork(
        tf_session=self.tf_session,
        state_size=settings.state_dim,
        action_size=3,
        tau=settings.tau,
        lr=settings.lrc
      )
    else:
      self.actor = ActorNetwork_CNN(
        tf_session=self.tf_session,
        tau=settings.tau,
        lr=settings.lra
      )
      self.critic = CriticNetwork_CNN(
        tf_session=self.tf_session,
        tau=settings.tau,
        lr=settings.lrc
      )

    try:
      self.actor.model.load_weights(settings.actor_weights_file)
      self.critic.model.load_weights(settings.critic_weights_file)
      print("Weights loaded successfully")
    except Exception as e:
      print(f"Weights loading failed: {str(e)}")

  def run(self, train_indicator):
    ep_rewards = []
    for episode in range(settings.episodes_num):
      print(f"Starting episode {episode}")

      # Get initial state
      if settings.WORKING_MODE in [settings.WORKING_MODE_OPTIONS[0], settings.WORKING_MODE_OPTIONS[1],
                                   settings.WORKING_MODE_OPTIONS[8]]:
        _, state = self.env.reset()
      else:
        current_state, _ = self.env.reset()
        if settings.IM_LAYERS == 1:
          current_state = np.expand_dims(current_state, -1)
        current_state = current_state / 255
        if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7]:
          current_state = current_state.flatten()

      total_reward = 0
      for step in range(settings.max_steps):
        # Get lane data from bridge process
        lane_data = None
        tm1 = time.time()

        loss = 0
        if not self.q.empty():
          lane_data = self.q.get_nowait()
          print(f"Using lane data: {lane_data}")

        # Select action
        if settings.WORKING_MODE in [settings.WORKING_MODE_OPTIONS[0], settings.WORKING_MODE_OPTIONS[1],
                                     settings.WORKING_MODE_OPTIONS[8]]:
          # action = self.actor.model.predict(state.reshape(1, state.shape[0]))[0]
          # new_state, reward, done, _ = self.env.step(action)
          # self.buffer.add((state, action, reward, new_state, done))
          # state = new_state

          action_predicted = self.actor.model.predict(state.reshape(1, state.shape[0]))  # + ou()  # predict and add noise
          [new_image, new_state], reward, done, info = self.env.step(action_predicted[0])
          self.buffer.add((state, action_predicted[0], reward, new_state, done))  # add replay buffer
          print(new_state)
          state = new_state
        else:
          action = self.actor.model.predict(np.array(current_state).reshape(-1, *current_state.shape))[0]
          new_current_state, reward, done, _ = self.env.step(action)
          if settings.IM_LAYERS == 1:
            new_current_state = np.expand_dims(new_current_state, -1)
          new_current_state = new_current_state / 255
          if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7]:
            new_current_state = new_current_state.flatten()
          self.buffer.add((current_state, action, reward, new_current_state, done))
          current_state = new_current_state

        total_reward += reward

        # Train networks
        if train_indicator and len(self.buffer) > settings.batch_size:
          self.train_step(loss)

        print("Episode %s - Step %s - Action %s - Reward %s" % (episode, step, action_predicted[0], reward))
        self.step += 1

        if done:
          print(self.env.summary)
          print("Episode %s - Step %s - Action %s - Reward %s" % (episode, step, action_predicted[0], reward))
          break

      ep_rewards.append(total_reward)
      if (episode > 0) and ((episode % AGGREGATE_STATS_EVERY == 0) or (episode == 1)):
        average_reward = np.mean(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        average_dist = np.mean(self.env.distance_acum[-AGGREGATE_STATS_EVERY:])
        # tensorboard.update_stats(average_reward=average_reward, min_reward=min_reward, max_reward=max_reward,
        #                          distance=average_dist, loss=loss)
      if episode % 3 == 0 and train_indicator:
        self.actor.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_actor.h5", overwrite=True)
        self.critic.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_critic.h5", overwrite=True)
      if episode % settings.N_save_stats == 0 and train_indicator:
        self.actor.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_" + str(episode) + "_actor.h5",
                                 overwrite=True)
        self.critic.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_" + str(episode) + "_critic.h5",
                                  overwrite=True)
      if (episode > 10) and (total_reward > np.max(ep_rewards[:-1])):
        self.actor.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_best_reward_actor.h5",
                                 overwrite=True)
        self.critic.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_best_reward_critic.h5",
                                  overwrite=True)

      time_buff.append((time.time() - tm1))
      # print(np.mean(np.array(time_buff)))
      tm = time.strftime("%Y-%m-%d %H:%M:%S")
      episode_stat = "%s -th Episode. %s total steps. Total reward: %s. Time %s" % (episode, self.step, total_reward, tm)
      dif_time = "Step time %s" % (time.time() - tm1)
      print(episode_stat)

      # Guardar estadísticas de cada episode
      # with open(train_stat_file, "a") as outfile:
      #     outfile.write(episode_stat + "\n")
      for actor_world in self.env.actor_list:
        actor_world.destroy()

    self.actor.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_actor.h5", overwrite=True)
    self.critic.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_critic.h5", overwrite=True)


  def train_step(self, loss):
    batch = self.buffer.get_batch(settings.batch_size)
    # states = np.asarray([e[0] for e in batch])
    # actions = np.asarray([e[1] for e in batch])
    # rewards = np.asarray([e[2] for e in batch])
    # new_states = np.asarray([e[3] for e in batch])
    # dones = np.asarray([e[4] for e in batch])
    states = tf.convert_to_tensor([e[0] for e in batch], dtype=tf.float32)
    actions = tf.convert_to_tensor([e[1] for e in batch], dtype=tf.float32)
    rewards = tf.convert_to_tensor([e[2] for e in batch], dtype=tf.float32)
    new_states = tf.convert_to_tensor([e[3] for e in batch], dtype=tf.float32)
    dones = tf.convert_to_tensor([e[4] for e in batch], dtype=tf.float32)

    target_q = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])
    y = rewards + settings.gamma * target_q.flatten() * (1 - dones)

    # Update critic
    loss += self.critic.model.train_on_batch([states, actions], y)

    # Update actor
    a_for_grad = self.actor.model.predict(states)
    a_for_grad = tf.convert_to_tensor(a_for_grad, dtype=tf.float32)
    grads = self.critic.get_gradients(states, a_for_grad)
    self.actor.train(states, grads)

    # Update target networks
    self.actor.train_target_model()
    self.critic.train_target_model()


def start_bridge_process(q):
  import sys
  sys.path.append("/home/yanxi/carla/openpilot_0.8.9/openpilot0.8.9")
  sys.path.append("/home/yanxi/carla/openpilot_0.8.9/openpilot0.8.9/tools/sim")
  from bridge_for_parallel import bridge
  Radar_Point_Array = Array('d', np.array([0] * (64 * 4 + 16 * 3 + 1)))
  N_radar_points = Value('i', 0)

  # generate vehicle
  client = carla.Client("localhost", 2000)
  world = client.get_world()
  blueprint_library = world.get_blueprint_library()
  model_3 = blueprint_library.filter("model3")[0]
  model_3.set_attribute('role_name', 'init')
  transform = random.choice(world.get_map().get_spawn_points())
  vehicle = world.spawn_actor(model_3, transform)

  bridge_proc = Process(
    target=bridge,
    args=(q, Radar_Point_Array, N_radar_points),
    daemon=True
  )
  bridge_proc.start()
  return bridge_proc


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-t", "--train", type=int, default=1)
  args = parser.parse_args()

  # Create communication queue
  q = Queue(maxsize=10)  # Limit queue size to prevent memory issues

  # Start training process
  trainer = DDPGTrainer(q)

  # Start bridge process after TF initialization is complete
  bridge_proc = start_bridge_process(q)

  try:
    # Run training
    trainer.run(args.train)
  except KeyboardInterrupt:
    print("Training interrupted")
  finally:
    bridge_proc.terminate()
    bridge_proc.join()
    print("Processes terminated")
