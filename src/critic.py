import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kbckend

from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, concatenate, Flatten, BatchNormalization, Activation

LAYER1_SIZE = 400
LAYER2_SIZE = 300

class Critic:
    def __init__(self, states_dim, actions_dim, lr, tau):
        self.states_dim = states_dim
        self.actions_dim = actions_dim
        self.lr = lr
        self.tau = tau

        self.model = self.get_model()
        self.target_model = self.get_model()

        # self.action_gradients = kbckend.function([self.model.input[0], self.model.input[1]], kbckend.gradients(self.model.output, [self.model.input[1]]))
    
    def get_model(self):
        state_inputs = Input((self.states_dim))
        action_inputs = Input((self.actions_dim))
        
        # layer1 = Dense(LAYER1_SIZE, activation = 'relu')(state_inputs)
        layer1 = Dense(LAYER1_SIZE)(state_inputs)
        layer1 = BatchNormalization()(layer1)
        layer1 = Activation("relu")(layer1)

        # layer2 = Dense(LAYER2_SIZE, activation = 'relu')(concatenate([Flatten()(layer1), action_inputs]))
        layer2 = Dense(LAYER2_SIZE)(concatenate([Flatten()(layer1), action_inputs]))
        layer2 = BatchNormalization()(layer2)
        layer2 = Activation("relu")(layer2)

        outputs = Dense(1, activation = 'linear', kernel_initializer = RandomUniform(-3e-3, 3e-3))(layer2)
        model = Model([state_inputs, action_inputs], outputs)
        model.compile(Adam(self.lr), 'mse')

        return model
    
    # def get_action_gradients(self, states, actions):
    #     return self.action_gradients([states, actions])
    
    def evaluate_action(self, state, action):
        return self.model.predict([state, action])

    def evaluate_action_target(self, state, action):
        return self.target_model.predict([state, action])
    
    def train(self, states, actions, Q_targets):
        self.model.train_on_batch([states, actions], Q_targets)
    
    # def update_target_model(self):
    #     weights = self.model.get_weights()
    #     target_weights = self.target_model.get_weights()
    #     for i in range(len(weights)):
    #         target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
    #     self.target_model.set_weights(target_weights)

    def update_target_model(self):
        # weights = self.model.get_weights()
        source_variables = self.model.weights
        target_variables = self.target_model.weights

        # for i in range(len(weights)):
        #     target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        # self.target_model.set_weights(target_weights)

        def update_op(target_variable, source_variable, tau):
            return target_variable.assign(
                tau * source_variable + (1.0 - tau) * target_variable, False)

        # with tf.name_scope(name, values=target_variables + source_variables):
        update_ops = [update_op(target_var, source_var, self.tau)
                    for target_var, source_var
                    in zip(target_variables, source_variables)]
        return tf.group(name="update_all_variables", *update_ops)
    
    def save_model(self, path):
        self.model.save_weights(path + '_actor.h5')

    def load_model(self, path):
        self.model.load_weights(path)
