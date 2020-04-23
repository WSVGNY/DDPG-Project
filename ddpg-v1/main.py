from openai_gym import LunarLanderGym
from agent import Agent
from gym.wrappers import Monitor
import tensorflow as tf
import datetime 
from keras.backend.tensorflow_backend import set_session
import os
import sys
from os import walk
import matplotlib.pyplot as plt
import argparse

# tf.compat.v1.disable_eager_execution()

def main():
    # For GPU, comment next line
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--gpu', type=str, default="-1", help='GPU ID')

    args = parser.parse_args(sys.argv[1:])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    critic_lr = args.lr

    set_session(tf.Session())
    env = LunarLanderGym().env
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    agent = Agent(state_dim, action_dim, buffer_size=1000000, minibatch_size=64, lr=critic_lr)
    scores = agent.train(env, render=False, nb_episodes=2000)

    if not os.path.exists("results"):
        os.mkdir("results")

    with open("results/ddpg-v1_latest_lr_{}.csv".format(critic_lr), "w") as f:
        for i in range(len(scores)):
            f.write(str(scores[i])[1:-1] + "\n")

    plt.plot([i + 1 for i in range(0, len(scores), 4)], scores[::4])
    plt.savefig("results/ddpg-v1_latest_lr_{}.png".format(critic_lr))

    env.close()

if __name__ == "__main__":
    main()
