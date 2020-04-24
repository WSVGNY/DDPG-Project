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
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--gpu', type=str, default="-1", help='GPU ID')

    args = parser.parse_args(sys.argv[1:])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    critic_lr = args.lr

    set_session(tf.Session())
    env = LunarLanderGym().env
    state_dim = env.observation_space.shape
    action_dim = env.action_space.high.shape[0]

    agent = Agent(state_dim, action_dim, buffer_size=1000000, minibatch_size=64, lr=critic_lr)
    scores = agent.train(env, render=False, nb_episodes=5000)

    if not os.path.exists("results"):
        os.mkdir("results")

    episode_score = []
    average_score = []
    with open("results/v1_latest_lr_{}.csv".format(critic_lr), "w") as f:
        for i in range(len(scores)):
            f.write(str(scores[i])[1:-1] + "\n")
            episode_score.append(scores[i][2])
            average_score.append(scores[i][3])

    plt.plot(list(range(len(scores))), episode_score, average_score)
    plt.savefig("results/v1_latest_lr_{}.png".format(critic_lr))

    env.close()

if __name__ == "__main__":
    main()
