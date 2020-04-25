from openai_gym import LunarLanderGym
from agent import Agent
from gym.wrappers import Monitor
import tensorflow as tf
import datetime 
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
    parser.add_argument('--ep', type=int, default=2000)
    parser.add_argument('--gpu', type=str, default="-1", help='GPU ID')

    args = parser.parse_args(sys.argv[1:])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    critic_lr = args.lr

    with tf.Session() as session:
        env = LunarLanderGym().env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        agent = Agent(session, state_dim, action_dim, buffer_size=1000000, minibatch_size=64, lr=critic_lr)
        scores = agent.train(env, render=False, nb_episodes=args.ep)

        if not os.path.exists("results"):
            os.mkdir("results")

        filename = "results/v2_lr-{}_ep-{}".format(critic_lr,args.ep)

        episode_score = []
        average_score = []
        with open(filename + ".csv", "w") as f:
            for i in range(len(scores)):
                f.write(str(scores[i])[1:-1] + "\n")
                episode_score.append(scores[i][2])
                average_score.append(scores[i][3])

        plt.plot(list(range(len(scores))), episode_score, average_score)
        plt.savefig(filename + ".png")

        env.close()

if __name__ == "__main__":
    main()
