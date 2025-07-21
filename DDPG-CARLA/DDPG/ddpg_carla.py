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
# from util.noise import OrnsteinUhlenbeckActionNoise
from DDPG.replay_buffer import ReplayBuffer
import cv2
import carla_config as settings
from keras.models import load_model
from typing import Any
import random
import carla

time_buff = []

AGGREGATE_STATS_EVERY = 10

class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._should_write_train = True

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs) if logs else None

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
        # self._write_logs(stats, self.step)
        with self.writer.as_default():
          for name, value in stats.items():
            tf.summary.scalar(name, value, step=self.step)




def play(train_indicator, q: Any):
    tf.keras.backend.clear_session()
    tf.config.experimental_run_functions_eagerly(True)  # 禁用图模式
    physical_devices = tf.config.list_physical_devices('GPU')
    [tf.config.experimental.set_memory_growth(dev, True) for dev in physical_devices]

    # ou_sigma = 0.3
    tensorboard = ModifiedTensorBoard(log_dir=f"logs/logs_{settings.WORKING_MODE}/{settings.TRAIN_MODE}-{int(time.time())}")
    step = 0

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf_session = tf.compat.v1.Session(config=config)

    tf.compat.v1.keras.backend.set_session(tf_session)

    print("yyx debug ddpg --- q is empty: ", q.empty())
    while not q.empty():
      message = q.get()
      print("receive lane lines from op: ", message)


    print("yyx test 1 --- working mode: ", settings.WORKING_MODE)
    if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1]\
            or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7] or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8]:
        print("yyx test 1.1##########")
        actor = ActorNetwork(tf_session=tf_session, state_size=settings.state_dim, action_size=3,
                             tau=settings.tau, lr=settings.lra)
        print("yyx test 1.2#######")
        critic = CriticNetwork(tf_session=tf_session, state_size=settings.state_dim, action_size=3,
                               tau=settings.tau, lr=settings.lrc)
        print("yyx test 1.3#######")
    else:
        actor = ActorNetwork_CNN(tf_session=tf_session, tau=settings.tau, lr=settings.lra)
        critic = CriticNetwork_CNN(tf_session=tf_session, tau=settings.tau, lr=settings.lrc)

    buffer = ReplayBuffer(settings.buffer_size)

    env = CarEnv()
    print("yyx test 2###########################")

    # noise function for exploration
    # ou = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim), sigma=ou_sigma * np.ones(action_dim))

    # Torcs environment - throttle and gear change controlled by client


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

    for i in range(settings.episodes_num):
        tensorboard.step = i
        print("Episode : %s Replay buffer %s" % (i, len(buffer)))
        if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1]\
                or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8]:
            _, state = env.reset()

        else:
            current_state, _ = env.reset()
            if settings.IM_LAYERS == 1:
                current_state = np.expand_dims(current_state, -1)
            current_state = current_state / 255
            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7]:
                current_state = current_state.flatten()

        if settings.GUARDAR_DATOS == 1:
        #Guardar los waypoints
            np.savetxt('Waypoints/DDPG_wp_' + str(settings.WORKING_MODE) + '_' + str(settings.TRAIN_MODE) + '_' + str(i) + '.txt', env.waypoints_txt, delimiter=';')

        total_reward = 0.0
        for j in range(settings.max_steps):
            tm1 = time.time()

            loss = 0
            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] \
                    or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8]:
                # sys.path.append("/home/yanxi/carla/openpilot_0.8.9/openpilot0.8.9")
                # sys.path.append("/home/yanxi/carla/openpilot_0.8.9/openpilot0.8.9/tools/sim")
                # from bridge_to_get_lane_bound_new import bridge
                # Radar_Point_Array = Array('d', np.array([0] * (64 * 4 + 16 * 3 + 1)).astype(
                #   'float64'))  # [0:64*4] radar points, [256:304]: can message 16*3, [304]: can message ready
                # N_radar_points = Value('i', 0)
                # bridge(q, Radar_Point_Array, N_radar_points, env.vehicle)
                #
                # if not q.empty():
                #   message = q.get()
                #   print("yyx debug ddpg receive message from op: ", message)
                action_predicted = actor.model.predict(state.reshape(1, state.shape[0]))  # + ou()  # predict and add noise
                [new_image, new_state], reward, done, info = env.step(action_predicted[0])
                buffer.add((state, action_predicted[0], reward, new_state, done))  # add replay buffer
                print(new_state)

            else:
                action_predicted = actor.model.predict(np.array(current_state).reshape(-1, *current_state.shape))  # + ou()  # predict and add noise
                [new_current_state, _], reward, done, info = env.step(action_predicted[0])
                if settings.IM_LAYERS == 1:
                    new_current_state = np.expand_dims(new_current_state, -1)
                new_current_state = new_current_state / 255
                if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7]:
                    new_current_state = new_current_state.flatten()
                buffer.add((current_state, action_predicted[0], reward, new_current_state, done))  # add replay buffer


            # batch update
            batch = buffer.get_batch(settings.batch_size)

            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.zeros((len(batch), 1))
            #try:
            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + settings.gamma * target_q_values[k]

            if train_indicator:
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.get_gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.train_target_model()
                critic.train_target_model()



            total_reward += reward

            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1]\
                    or settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8]:
                state = new_state
            # elif settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1]:
            #     state = state1
            else:
                current_state = new_current_state

            # Imprimir estadísticas cada step
            print("Episode %s - Step %s - Action %s - Reward %s" % (i, step, action_predicted[0], reward))

            step += 1
            if done:
                print(env.summary)
                # Imprimir estadisticas cada episode
                print("Episode %s - Step %s - Action %s - Reward %s" % (i, step, action_predicted[0], reward))
                break


        #Guardar datos en tensorboard
        ep_rewards.append(total_reward)
        if (i > 0) and ((i % AGGREGATE_STATS_EVERY == 0) or (i ==1)):
            average_reward = np.mean(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            average_dist = np.mean(env.distance_acum[-AGGREGATE_STATS_EVERY:])
            tensorboard.update_stats(average_reward=average_reward, min_reward=min_reward, max_reward=max_reward,
                                     distance=average_dist, loss=loss)

        #Guardar datos del entrenamiento en ficheros
        if i % 3 == 0 and train_indicator:
            actor.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_actor.h5", overwrite=True)
            critic.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_critic.h5", overwrite=True)
        if i % settings.N_save_stats == 0 and train_indicator:
            actor.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_" + str(i) + "_actor.h5", overwrite=True)
            critic.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_" + str(i) + "_critic.h5", overwrite=True)
        if (i > 10) and (total_reward > np.max(ep_rewards[:-1])):
            actor.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_best_reward_actor.h5", overwrite=True)
            critic.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_best_reward_critic.h5", overwrite=True)

        if settings.GUARDAR_DATOS == 1:
        #Guardar la trayectoria recorrida
            np.savetxt('Trayectorias/DDPG_trayectoria_' + str(settings.WORKING_MODE) + '_' + str(settings.TRAIN_MODE) + '_' + str(i) + '.txt', env.position_array, delimiter=';')
            env.position_array = []

        time_buff.append((time.time() - tm1))
        #print(np.mean(np.array(time_buff)))
        tm = time.strftime("%Y-%m-%d %H:%M:%S")
        episode_stat = "%s -th Episode. %s total steps. Total reward: %s. Time %s" % (i, step, total_reward, tm)
        dif_time = "Step time %s" % (time.time() - tm1)
        print(episode_stat)

        # Guardar estadísticas de cada episode
        # with open(train_stat_file, "a") as outfile:
        #     outfile.write(episode_stat + "\n")
        for actor_world in env.actor_list:
            actor_world.destroy()

    actor.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_actor.h5", overwrite=True)
    critic.model.save_weights(settings.save_weights_path + str(settings.TRAIN_MODE) + "_critic.h5", overwrite=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", type=int, help="train indicator", default=1)
    args = parser.parse_args()

    # generate vehicle
    client = carla.Client("localhost", 2000)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    model_3 = blueprint_library.filter("model3")[0]
    model_3.set_attribute('role_name', 'init')
    transform = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(model_3, transform)

    # from multiprocessing import Queue, Pipe, Process, Array, Value
    # from typing import Any

    # q: Any = Queue()
    # play(args.train, q)
    from multiprocessing import Queue, Pipe, Process, Array, Value
    from typing import Any

    q: Any = Queue()
    p = Process(target=play, args=(args.train, q), daemon=True)
    p.start()
    # play(args.train, q)

    sys.path.append("/home/yanxi/carla/openpilot_0.8.9/openpilot0.8.9")
    sys.path.append("/home/yanxi/carla/openpilot_0.8.9/openpilot0.8.9/tools/sim")
    from bridge_to_get_lane_bound_new import bridge_keep_alive, bridge

    Radar_Point_Array = Array('d', np.array([0] * (64 * 4 + 16 * 3 + 1)).astype(
      'float64'))  # [0:64*4] radar points, [256:304]: can message 16*3, [304]: can message ready
    N_radar_points = Value('i', 0)
    bridge(q, Radar_Point_Array, N_radar_points)
    # bridge_proc = Process(target=bridge, args=(q, Radar_Point_Array, N_radar_points, world), daemon=True)
    # bridge_proc.start()
    # time.sleep(5.0)

    #
    # play_proc = Process(target=play, args=(args.train, q), daemon=True)
    # play_proc.start()

    # p1 = Process(target=play, args=(args.train, q), daemon=True)
    # p2 = Process(target=bridge, args=(q, Radar_Point_Array, N_radar_points), daemon=True)
    #
    # p1.start()
    # p2.start()

    # try:
    #   while True:
    #     time.sleep(1)
    # except KeyboardInterrupt:
    #   print("Shutting down")
    #   p1.join()
    #   p2.join()

    # vehicle.destroy()
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
