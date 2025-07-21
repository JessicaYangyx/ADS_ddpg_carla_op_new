import keras.backend as keras_backend
import numpy as np
import tensorflow as tf
from click.core import batch
from tensorflow.keras.layers import Dense, Input, Add, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from carla_config import hidden_units, image_network


class CriticNetwork:
    def __init__(self, tf_session, state_size, action_size=1, tau=0.001, lr=0.001):
        # self.tf_session = tf_session
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.lr = lr

        # tf.compat.v1.keras.backend.set_session(tf_session)

        self.model, self.state_input, self.action_input = self.generate_model()

        self.target_model, _, _ = self.generate_model()
        self.target_model.set_weights(self.model.get_weights())

        # self.critic_gradients = tf.gradients(self.model.output, self.action_input)
        # self.tf_session.run(tf.global_variables_initializer())
        self.critic_gradients = self.get_gradients_fn()

    def get_gradients_fn(self):
      @tf.function
      def grad_fn(states, actions):
        with tf.GradientTape() as tape:
          tape.watch(actions)
          q_values = self.model([states,actions])
        return tape.gradient(q_values, actions)
      return grad_fn

    def get_gradients(self, states, actions):
        # return self.tf_session.run(
        #     self.critic_gradients,
        #     feed_dict={self.state_input: states, self.action_input: actions},
        # )[0]
        return self.critic_gradients(states, actions).numpy()

    def train_target_model(self):
        main_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        target_weights = [
            self.tau * main_weight + (1 - self.tau) * target_weight
            for main_weight, target_weight in zip(main_weights, target_weights)
        ]
        self.target_model.set_weights(target_weights)

    def generate_model(self):
        state_input = Input(shape=[self.state_size])
        state_h1 = Dense(hidden_units[0], activation="relu")(state_input)
        state_h2 = Dense(hidden_units[1], activation="linear")(state_h1)

        action_input = Input(shape=[self.action_size])
        action_h1 = Dense(hidden_units[1], activation="linear")(action_input)

        merged = Add(name='merge')([state_h2, action_h1])
        merged_h1 = Dense(hidden_units[1], activation="relu")(merged)

        output_layer = Dense(1, activation="linear")(merged_h1)
        model = Model(inputs=[state_input, action_input], outputs=output_layer)




        model.compile(loss="mse", optimizer=Adam(lr=self.lr))
        tf.keras.utils.plot_model(model, to_file=image_network + 'critic_model_WP_Carla.png',
                                  show_shapes=True, show_layer_names=True, rankdir='TB')
        return model, state_input, action_input
