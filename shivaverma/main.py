import gym
import tflearn
import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from noise import OUNoise
from actor import ActorNetwork
from critic import CriticNetwork
from replay_buffer import ReplayBuffer
import argparse
import time as time


def train(sess, env, actor, critic, actor_noise, buffer_size, min_batch, ep):

    sess.run(tf.global_variables_initializer())

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(buffer_size, 0)

    max_episodes = ep
    max_steps = 3000
    score_list = []
    start_time = time.time()

    for i in range(max_episodes):

        state = env.reset()
        score = 0

        for j in range(max_steps):

            # env.render()

            action = actor.predict(np.reshape(state, (1, actor.s_dim))) + actor_noise()
            next_state, reward, done, info = env.step(action[0])
            replay_buffer.add(np.reshape(state, (actor.s_dim,)), np.reshape(action, (actor.a_dim,)), reward,
                              done, np.reshape(next_state, (actor.s_dim,)))

            # updating the network in batch
            if replay_buffer.size() < min_batch:
                continue

            states, actions, rewards, dones, next_states = replay_buffer.sample_batch(min_batch)
            target_q = critic.predict_target(next_states, actor.predict_target(next_states))

            y = []
            for k in range(min_batch):
                y.append(rewards[k] + critic.gamma * target_q[k] * (1-dones[k]))

            # Update the critic given the targets
            predicted_q_value, _ = critic.train(states, actions, np.reshape(y, (min_batch, 1)))

            # Update the actor policy using the sampled gradient
            a_outs = actor.predict(states)
            grads = critic.action_gradients(states, a_outs)
            actor.train(states, grads[0])

            # Update target networks
            actor.update_target_network()
            critic.update_target_network()

            state = next_state
            score += reward

            if done:
                break
        
        avg = np.mean([s[2] for s in score_list[-99:]] + [score])
        score_list.append((i, time.time() - start_time, score, avg))
        print(str(score_list[-1])[1:-1])

        if avg > 200:
            print('Task completed in {}'.format(time.time() - start_time))
            break

    return score_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training parameters')    
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--ep', type=int, default=2000)
    parser.add_argument('--gpu', type=str, default="-1", help='GPU ID')

    args = parser.parse_args(sys.argv[1:])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    critic_lr = args.lr

    with tf.Session() as sess:
        env = gym.make('LunarLanderContinuous-v2')

        env.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)

        ep = args.ep
        tau = 0.001
        gamma = 0.99
        min_batch = 64
        actor_lr = critic_lr * 0.1
        buffer_size = 1000000

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high

        actor_noise = OUNoise(mu=np.zeros(action_dim))
        actor = ActorNetwork(sess, state_dim, action_dim, action_bound, actor_lr, tau, min_batch)
        critic = CriticNetwork(sess, state_dim, action_dim, critic_lr, tau, gamma, actor.get_num_trainable_vars())
        scores = train(sess, env, actor, critic, actor_noise, buffer_size, min_batch, ep)


        if not os.path.exists("results"):
            os.mkdir("results")

        filename = "results/shiv_lr-{}_ep-{}".format(critic_lr,args.ep)

        episode_score = []
        average_score = []
        with open(filename + ".csv", "w") as f:
            for i in range(len(scores)):
                f.write(str(scores[i])[1:-1] + "\n")
                episode_score.append(scores[i][2])
                average_score.append(scores[i][3])

        plt.plot(list(range(len(scores))), episode_score, average_score)
        plt.savefig(filename + ".png")

