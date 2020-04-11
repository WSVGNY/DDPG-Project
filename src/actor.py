import numpy as np
import tensorflow as tf
import keras.backend as kbckend

from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Input, Dense

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
        layer1 = Dense(LAYER1_SIZE, activation = 'relu')(inputs)
        layer2 = Dense(LAYER2_SIZE, activation = 'relu')(layer1)
        outputs = Dense(self.outputs_dim, activation = 'tanh', kernel_initializer = RandomUniform(-3e-3, 3e-3))(layer2)

        return Model(inputs, outputs)
    
    # C'est sketch mais je trouve pas de meilleure alternative ...
    def get_optimize(self):
        outputs_gradients = kbckend.placeholder(shape = (None, self.outputs_dim))
        params_gradients = tf.gradients(self.model.output, self.model.trainable_weights, -outputs_gradients)
        gradients = zip(params_gradients, self.model.trainable_weights)

        return kbckend.function([self.model.input, outputs_gradients], [tf.optimizers.Adam(self.lr).apply_gradients(gradients)])
    
    def choose_action(self, state):
        return self.model.predict(state)

    def choose_action_target(self, state):
        return self.target_model.predict(state)
    
    def train(self, states, gradients):
        self.optimize([states, gradients])
    
    def update_target_model(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        [target_weights[i].assign(self.tau * weights[i] + (1 - self.tau) * target_weights[i]) for i in range(len(weights))]
        self.target_model.set_weights(target_weights)
