""" Deep RL Algorithms for OpenAI Gym environments
"""

import os
import sys
import gym
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from ddpg import DDPG

from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical

from continuous_environments import Environment
from networks import get_session
import matplotlib.pyplot as plt

gym.logger.set_level(40)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    # parser.add_argument('--type', type=str, default='DDQN',help="Algorithm to train from {A2C, A3C, DDQN, DDPG}")
    # parser.add_argument('--is_atari', dest='is_atari', action='store_true', help="Atari Environment")
    # parser.add_argument('--with_PER', dest='with_per', action='store_true', help="Use Prioritized Experience Replay (DDQN + PER)")
    # parser.add_argument('--dueling', dest='dueling', action='store_true', help="Use a Dueling Architecture (DDQN)")
    # #
    # parser.add_argument('--nb_episodes', type=int, default=5000, help="Number of training episodes")
    # parser.add_argument('--batch_size', type=int, default=64, help="Batch size (experience replay)")
    # parser.add_argument('--consecutive_frames', type=int, default=4, help="Number of consecutive frames (action repeat)")
    # parser.add_argument('--training_interval', type=int, default=30, help="Network training frequency")
    # parser.add_argument('--n_threads', type=int, default=8, help="Number of threads (A3C)")
    # #
    # parser.add_argument('--gather_stats', dest='gather_stats', action='store_true',help="Compute Average reward per episode (slower)")
    # parser.add_argument('--render', dest='render', action='store_true', help="Render environment while training")
    # parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4',help="OpenAI Gym Environment")
    # parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    #
    # parser.set_defaults(render=False)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--gpu', type=str, default="-1", help='GPU ID')

    return parser.parse_args(args)

def main(args=None):

    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    critic_lr = args.lr

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_session(get_session())

    env = Environment(gym.make('LunarLanderContinuous-v2'))
    env.reset()
    state_dim = env.get_state_size()
    action_space = gym.make('LunarLanderContinuous-v2').action_space
    action_dim = action_space.high.shape[0]
    act_range = action_space.high

    # Pick algorithm to train
    algo = DDPG(action_dim, state_dim, act_range, lr=critic_lr, buffer_size=1000000)

    # Train
    scores = algo.train(env, nb_episodes=2000)


    if not os.path.exists("results"):
        os.mkdir("results")

    episode_score = []
    average_score = []
    with open("results/germain_latest_lr_{}.csv".format(critic_lr), "w") as f:
        for i in range(len(scores)):
            f.write(str(scores[i])[1:-1] + "\n")
            episode_score.append(scores[i][2])
            average_score.append(scores[i][3])

    plt.plot(list(range(len(scores))), episode_score, average_score)
    plt.savefig("results/germain_latest_lr_{}.png".format(critic_lr))

    # Export results to CSV
    # if(args.gather_stats):
    #     df = pd.DataFrame(np.array(stats))
    #     df.to_csv(args.type + "/logs.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')

    # Save weights and close environments
    # exp_dir = '{}/models/'.format(args.type)
    # if not os.path.exists(exp_dir):
    #     os.makedirs(exp_dir)

    # export_path = '{}{}_ENV_{}_NB_EP_{}_BS_{}'.format(exp_dir,
    #     args.type,
    #     args.env,
    #     args.nb_episodes,
    #     args.batch_size)

    # algo.save_weights(export_path)
    env.env.close()

if __name__ == "__main__":
    main()
