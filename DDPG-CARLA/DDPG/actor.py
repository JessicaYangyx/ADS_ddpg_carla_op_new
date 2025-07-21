import keras.backend as keras_backend
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
from keras.initializers import normal
from keras.layers import Dense, Input
from keras.models import Model
from carla_config import hidden_units, image_network
from tensorflow import optimizers


class ActorNetwork:
    def __init__(self, tf_session, state_size, action_size=1, tau=0.001, lr=0.0001):
        self.tf_session = tf_session
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.lr = lr

        print("yyx in actor 1")
        tf.compat.v1.keras.backend.set_session(tf_session)
        print("yyx in actor 2")

        self.model, self.model_states = self.generate_model()
        model_weights = self.model.trainable_weights
        print("yyx in actor 3")

        self.target_model, _ = self.generate_model()
        self.target_model.set_weights(self.model.get_weights())
        print("yyx in actor 4")

        # Generate tensors to hold the gradients for Policy Gradient update
        # self.action_gradients = tf.compat.v1.placeholder(tf.float32, [None, action_size])
        # self.parameter_gradients = tf.gradients(self.model.output, model_weights, -self.action_gradients)
        # self.gradients = zip(self.parameter_gradients, model_weights)
        #
        # self.optimize = tf.compat.v1.train.AdamOptimizer(self.lr).apply_gradients(self.gradients)
        # self.tf_session.run(tf.compat.v1.global_variables_initializer())
        self.optimize = optimizers.Adam(learning_rate=self.lr)

        print("yyx in actor 5")

    def train(self, states, action_gradients):
        with tf.GradientTape() as tape:
          actions = self.model(states)
          grads = tape.gradient(actions, self.model.trainable_weights, output_gradients=-action_gradients)
        self.optimize.apply_gradients(zip(grads, self.model.trainable_weights))
        # self.tf_session.run(
        #     self.optimize,
        #     feed_dict={
        #         self.model_states: states,
        #         self.action_gradients: action_gradients,
        #     },
        # )

    def train_target_model(self):
        main_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        target_weights = [
            self.tau * main_weight + (1 - self.tau) * target_weight
            for main_weight, target_weight in zip(main_weights, target_weights)
        ]
        self.target_model.set_weights(target_weights)

    def generate_model(self):
        input_layer = tf.keras.layers.Input(shape=[self.state_size])
        print("yyx test in generate model 1")
        h0 = tf.keras.layers.Dense(hidden_units[0], activation="relu")(input_layer)
        print("yyx test in generate model 2")
        h1 = tf.keras.layers.Dense(hidden_units[1], activation="relu")(h0)
        print("yyx test in generate model 3")
        output_layer = tf.compat.v1.keras.layers.Dense(3, activation="tanh")(h1)
        print("yyx test in generate model 4")
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        print("yyx test in generate model 5")
        tf.keras.utils.plot_model(model,
                                  to_file=image_network + 'actor_model_WP_Carla.png',
                                  show_shapes=True,
                                  show_layer_names=True, rankdir='TB')
        print("yyx test in generate model 6")



        return model, input_layer
