import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kbckend

from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation

LAYER1_SIZE = 400
LAYER2_SIZE = 300

class Actor:
    def __init__(self, states_dim, actions_dim, lr, tau):
        self.inputs_dim = states_dim
        self.outputs_dim = actions_dim
        self.lr = lr
        self.tau = tau

        self.model = self.get_model()
        self.target_model = self.get_model()
        self.optimize = self.get_optimize()

    def get_model(self):
        inputs = Input((self.inputs_dim))
        
        # layer1 = Dense(LAYER1_SIZE, activation = 'relu')(inputs)
        layer1 = Dense(LAYER1_SIZE)(inputs)
        layer1 = BatchNormalization()(layer1)
        layer1 = Activation("relu")(layer1)
        
        # layer2 = Dense(LAYER2_SIZE, activation = 'relu')(layer1)
        layer2 = Dense(LAYER2_SIZE)(layer1)
        layer2 = BatchNormalization()(layer2)
        layer2 = Activation("relu")(layer2)
        outputs = Dense(self.outputs_dim, activation = 'tanh', kernel_initializer = RandomUniform(-3e-3, 3e-3))(layer2)
        model = Model(inputs, outputs)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        model.compile(optimizer=self.optimizer, loss='mse')

        return model
    
    # Train the actor network to maximise the Action - State pair's Q-value
    # outputs-gradients :   Gradient of the Action - State pair's Q-value from Critic with respect to the action
    # 
    #  state -> Actor -> [state, action] -> Critic -> Q-value
    def get_optimize(self):
        outputs_gradients = kbckend.placeholder(shape = (None, self.outputs_dim))
        params_gradients = tf.gradients(self.model.output, self.model.trainable_weights, -outputs_gradients)
        gradients = zip(params_gradients, self.model.trainable_weights)

        return kbckend.function(inputs=[self.model.input, outputs_gradients], outputs=[kbckend.constant(1)], updates=[tf.train.AdamOptimizer(self.lr).apply_gradients(gradients)][1:])
        # tf.train.AdamOptimizer(self.lr).apply_gradients(grads)

    def choose_action(self, state):
        return self.model.predict(state)

    def choose_action_target(self, state):
        return self.target_model.predict(state)
    
    # Backpropagation with gradient of the Action - State pair's Q-value from Critic with respect to the action
    def train(self, states, gradients):
        self.optimize([states, gradients])

    # def train(self, X_train, critic_grads):
    #     with tf.GradientTape() as tape:
    #         y_pred = self.model(X_train, training=True)
    #     actor_grads = tape.gradient(y_pred, self.model.trainable_variables, output_gradients=critic_grads*-1)
    #     self.optimizer.apply_gradients(zip(actor_grads, self.model.trainable_variables))
        #print(actor_grads)
    
    def update_target_model(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        # source_variables = self.model.weights
        # target_variables = self.target_model.weights

        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

        # def update_op(target_variable, source_variable, tau):
        #     return target_variable.assign(
        #         tau * source_variable + (1.0 - tau) * target_variable, False)

        # # with tf.name_scope(name, values=target_variables + source_variables):
        # update_ops = [update_op(target_var, source_var, self.tau)
        #             for target_var, source_var
        #             in zip(target_variables, source_variables)]
        # return tf.group(name="update_all_variables", *update_ops)
    
    def save_model(self, path):
        self.model.save_weights(path + '_actor.h5')

    def load_model(self, path):
        self.model.load_weights(path)
