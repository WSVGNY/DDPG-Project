from openai_gym import LunarLanderGym
from agent import Agent
from gym.wrappers import Monitor
import datetime 

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
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def main():
    env = LunarLanderGym().env
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    agent = Agent(state_dim, action_dim)
    
    agent.train(env, render=True)
    agent.evaluate(env, 10)

    env.close()

main()