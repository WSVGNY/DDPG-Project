import tensorflow as tf
import tflearn
from tflearn.initializations import uniform

LAYER1_SIZE = 400
LAYER2_SIZE = 200

class Actor:
    def __init__(self, session, states_dim, actions_dim, lr, tau, minibatch_size):
        self.session = session
        self.inputs_dim = states_dim
        self.outputs_dim = actions_dim
        self.lr = lr
        self.tau = tau
        self.minibatch_size = minibatch_size

        self.model = self.get_model()
        self.model["trainable_weights"] = tf.trainable_variables()
        self.target_model = self.get_model()
        self.target_model["trainable_weights"] = tf.trainable_variables()[len(self.model["trainable_weights"]):]
        self.optimize = self.get_optimize()
        self.update_target_network_params = \
            [self.target_model["trainable_weights"][i].assign(
                tf.multiply(self.model["trainable_weights"][i], self.tau) +
                tf.multiply(self.target_model["trainable_weights"][i], 1. - self.tau))
            for i in range(len(self.target_model["trainable_weights"]))]

        self.num_trainable_weights = len(self.model["trainable_weights"]) + len(self.target_model["trainable_weights"])

    def get_model(self):
        inputs = tflearn.input_data(shape=[None, self.inputs_dim])
        
        # layer1 = Dense(LAYER1_SIZE, activation = 'relu')(inputs)
        layer1 = tflearn.fully_connected(inputs, LAYER1_SIZE)
        layer1 = tflearn.layers.normalization.batch_normalization(layer1)
        layer1 = tflearn.activations.relu(layer1)
        
        # layer2 = Dense(LAYER2_SIZE, activation = 'relu')(layer1)
        layer2 = tflearn.fully_connected(layer1, LAYER2_SIZE, weights_init=uniform(minval=-0.002, maxval=0.002))
        layer2 = tflearn.layers.normalization.batch_normalization(layer2)
        layer2 = tflearn.activations.relu(layer2)
        outputs = tflearn.fully_connected(layer2, self.outputs_dim, activation = 'tanh', weights_init=uniform(minval=-0.004, maxval=0.004))

        return {"inputs": inputs, "outputs": outputs}
    
    # Train the actor network to maximise the Action - State pair's Q-value
    # outputs-gradients :   Gradient of the Action - State pair's Q-value from Critic with respect to the action
    # 
    #  state -> Actor -> [state, action] -> Critic -> Q-value
    def get_optimize(self):
        self.outputs_gradients = tf.placeholder(tf.float32, [None, self.outputs_dim])
        params_gradients = tf.gradients(self.model["outputs"], self.model["trainable_weights"], -self.outputs_gradients)
        gradients = list(map(lambda x: tf.div(x, self.minibatch_size), params_gradients))

        return tf.train.AdamOptimizer(self.lr).apply_gradients(zip(gradients, self.model["trainable_weights"]))

    def choose_action(self, state):
        return self.session.run(self.model["outputs"], feed_dict={
            self.model["inputs"]: state
        })

    def choose_action_target(self, state):
        return self.session.run(self.target_model["outputs"], feed_dict={
            self.target_model["inputs"]: state
        })
    
    # Backpropagation with gradient of the Action - State pair's Q-value from Critic with respect to the action
    def train(self, states, gradients):
        self.session.run(self.optimize, feed_dict={
            self.model["inputs"]: states,
            self.outputs_gradients: gradients
        })
    
    def update_target_model(self):
        self.session.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_weights