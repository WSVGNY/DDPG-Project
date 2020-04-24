import tensorflow as tf
import tflearn
from tflearn.initializations import uniform

LAYER1_SIZE = 400
LAYER2_SIZE = 200

class Critic:
    def __init__(self, session, states_dim, actions_dim, lr, tau, num_actor_trainable_weights):
        self.session = session
        self.states_dim = states_dim
        self.actions_dim = actions_dim
        self.lr = lr
        self.tau = tau

        self.model = self.get_model()
        self.model["trainable_weights"] = tf.trainable_variables()[num_actor_trainable_weights:]
        self.target_model = self.get_model()
        self.target_model["trainable_weights"] = tf.trainable_variables()[(len(self.model["trainable_weights"]) + num_actor_trainable_weights):]

        self.action_gradients = tf.gradients(self.model["outputs"], self.model["actions"])
        self.optimize = self.get_optimize()

        self.update_target_network_params = \
            [self.target_model["trainable_weights"][i].assign(
                tf.multiply(self.model["trainable_weights"][i], self.tau) +
                tf.multiply(self.target_model["trainable_weights"][i], 1. - self.tau))
            for i in range(len(self.target_model["trainable_weights"]))]

    def get_model(self):
        inputs = tflearn.input_data(shape=[None, self.states_dim])
        actions = tflearn.input_data(shape=[None, self.actions_dim])
        
        # layer1 = Dense(LAYER1_SIZE, activation = 'relu')(inputs)
        layer1 = tflearn.fully_connected(inputs, LAYER1_SIZE)
        layer1 = tflearn.layers.normalization.batch_normalization(layer1)
        layer1 = tflearn.activations.relu(layer1)
        
        # layer2 = Dense(LAYER2_SIZE, activation = 'relu')(layer1)
        layer2 = tflearn.fully_connected(layer1, LAYER2_SIZE, weights_init=uniform(minval=-0.002, maxval=0.002))
        layer2_action = tflearn.fully_connected(actions, LAYER2_SIZE)

        layer2 = tflearn.activation(tf.matmul(layer1, layer2.W) + tf.matmul(actions, layer2_action.W) + layer2_action.b, activation='relu')
        outputs = tflearn.fully_connected(layer2, 1, weights_init=uniform(minval=-0.004, maxval=0.004))

        return {"inputs": inputs, "outputs": outputs, "actions": actions}
    
    def get_optimize(self):
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
        loss = tflearn.mean_square(self.predicted_q_value, self.model["outputs"])

        return tf.train.AdamOptimizer(self.lr).minimize(loss)

    def get_action_gradients(self, states, actions):
        return self.session.run(self.action_gradients, feed_dict={
            self.model["inputs"]: states,
            self.model["actions"]: actions
        })
    
    def evaluate_action(self, state, action):
        return self.session.run(self.model["outputs"], feed_dict={
            self.model["inputs"]: state,
            self.model["actions"]: action
        })

    def evaluate_action_target(self, state, action):
        return self.session.run(self.target_model["outputs"], feed_dict={
            self.target_model["inputs"]: state,
            self.target_model["actions"]: action
        })
    
    def train(self, states, actions, Q_targets):
        return self.session.run([self.model["outputs"], self.optimize], feed_dict={
            self.model["inputs"]: states,
            self.model["actions"]: actions,
            self.predicted_q_value: Q_targets
        })
    
    def update_target_model(self):
        self.session.run(self.update_target_network_params)
