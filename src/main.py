from openai_gym import LunarLanderGym
from agent import Agent
from gym.wrappers import Monitor
import tensorflow as tf
import datetime 
from keras.backend.tensorflow_backend import set_session
import os
from os import walk



ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
MAX_EPISODES = 100000
MAX_STEPS_EPISODE = 50000
EXPLORATION_EPISODES = 10000
GAMMA = 0.99
TAU = 0.001
BUFFER_SIZE = 1000000
OU_THETA = 0.15
OU_MU = 0.
OU_SIGMA = 0.3
MIN_EPSILON = 0.1
MAX_EPSILON = 1
EVAL_PERIODS = 100
EVAL_EPISODES = 10

DATETIME = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
MONITOR_DIR = f'./results/${DATETIME}/gym_ddpg'

# SUMMARY_DIR = f'./results/${DATETIME}/tf_ddpg'
tf.compat.v1.disable_eager_execution()

SAVED_MODELS_PATH = "./saved_models/"

def main():
    # For GPU, comment next line
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    env = LunarLanderGym().env
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    agent = Agent(state_dim, action_dim, buffer_size=512)
    start = 0
    if not os.path.exists(SAVED_MODELS_PATH):
        os.mkdir(SAVED_MODELS_PATH)
    # else:
    #     (_, _, filenames) = walk(SAVED_MODELS_PATH).next()
    #     for f in filenames:
    #         start = max(start, int(f.split("_")[0]))
    #     agent.load("{}{}_actor".format(SAVED_MODELS_PATH, start), "{}{}_critic".format(SAVED_MODELS_PATH, start))

    for i in range(start, 5000, 10):
        agent.train(env, render=False, nb_episodes=10, loaded_episode=i)
        agent.evaluate(env, 10, render=True)

    env.close()

main()
