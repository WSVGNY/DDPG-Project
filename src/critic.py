import numpy as np
import tensorflow as tf
import keras.backend as kbckend

from keras.initializers import RandomUniform
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, concatenate, Flatten

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

        self.action_gradients = kbckend.function([self.model.input[0], self.model.input[1]], kbckend.gradients(self.model.output, [self.model.input[1]]))
    
    def get_model(self):
        state_inputs = Input((self.states_dim))
        action_inputs = Input((self.actions_dim))
        layer1 = Dense(LAYER1_SIZE, activation = 'relu')(state_inputs)
        layer2 = Dense(LAYER2_SIZE, activation = 'relu')(concatenate([Flatten()(layer1), action_inputs]))
        outputs = Dense(1, activation = 'linear', kernel_initializer = RandomUniform(-3e-3, 3e-3))(layer2)

        return Model([state_inputs, action_inputs], outputs).compile(Adam(self.lr), 'mse')
    
    def get_action_gradients(self, states, actions):
        return self.action_gradients([states, actions])
    
    def evaluate_action(self, state, action):
        return self.model.predict([state, action])

    def evaluate_action_target(self, state, action):
        return self.target_model.predict([state, action])
    
    def train(self, states, actions, Q_targets):
        self.model.train_on_batch([states, actions], Q_targets)
    
    def update_target_model(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        [target_weights[i].assign(self.tau * weights[i] + (1 - self.tau) * target_weights[i]) for i in range(len(weights))]
        self.target_model.set_weights(target_weights)
