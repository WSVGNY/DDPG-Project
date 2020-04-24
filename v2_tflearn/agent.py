import numpy as np
from actor import Actor
from critic import Critic
from utils.actionnoise import OrnsteinUhlenbeckProcess
from utils.replaybuffer import ReplayBuffer
import tensorflow as tf
from datetime import datetime
import time

class Agent:
    def __init__(self, session, states_dim, actions_dim, buffer_size = 20000, gamma = 0.99, lr = 0.0005, tau = 0.001, minibatch_size=64):
        self.session = session
        self.states_dim = states_dim
        self.actions_dim = actions_dim
        self.gamma = gamma
        self.lr = lr
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.minibatch_size = minibatch_size
        self.actor = Actor(session, self.states_dim, self.actions_dim, 0.1 * lr, tau, minibatch_size)
        self.critic = Critic(session, self.states_dim, self.actions_dim, lr, tau, self.actor.get_num_trainable_vars())

    def bellman(self, rewards, dones, next_state_rewards):
        target_rewards = np.asarray(next_state_rewards)
        for i in range(self.minibatch_size):
            if dones[i]:
                target_rewards[i] = rewards[i] 
            else:
                target_rewards[i] = rewards[i] + self.gamma * next_state_rewards[i]
        
        return target_rewards

    def train(self, env, nb_episodes=5000, render=False):
        self.session.run(tf.global_variables_initializer())
        score_list = []
        for i in range(nb_episodes):
            # Reinitialiser le jeu
            state = env.reset()
            episode_reward = 0
            done = False
            noise = OrnsteinUhlenbeckProcess(size = self.actions_dim)
            step = 0

            while not done:
                if render: env.render()
                action = np.clip(self.actor.choose_action(np.expand_dims(state, 0))[0] + noise.generate(step), -1, 1) 
                next_state, reward, done, _ = env.step(action)
                self.replay_buffer.store(state, action, reward, done, next_state)
                
                state = next_state
                episode_reward += reward
                # print("ep {} step #{} : done in {} s, reward = {}, buffer_size = {}/{}".format(i, step, time.time() - start_time, reward, self.replay_buffer.get_size(), self.replay_buffer.buffer_size))
                start_time = time.time()
                step += 1
        
                if self.replay_buffer.get_size() > self.minibatch_size:
                    states, actions, rewards, dones, next_states = self.replay_buffer.sample(self.minibatch_size)

                    #q_next
                    next_state_rewards = self.critic.evaluate_action_target(next_states, self.actor.choose_action_target(next_states))
                    target_rewards = self.bellman(rewards, dones, next_state_rewards)

                    # Update models
                    self.critic.train(states, actions, target_rewards)
                    training_actions = self.actor.choose_action(states)
                    training_actions_gradients = self.critic.get_action_gradients(states, training_actions)
                    self.actor.train(states, training_actions_gradients[0])

                    # Update target models
                    self.actor.update_target_model()
                    self.critic.update_target_model()
            
            avg = np.mean([s[2] for s in score_list[-99:]] + [episode_reward])
            score_list.append((i, time.time(), episode_reward, avg))
            print(str(score_list[-1])[1:-1])

            if avg > 200:
                print('Task Completed')
                break
        return score_list

    def evaluate(self, env, nb_episodes, render=False):
        scores = []
        for _ in range(nb_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                if render: env.render()
                action = self.actor.choose_action(np.expand_dims(state, 0))[0]
                # action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)                
                state = next_state
                episode_reward += reward
            
            scores.append(episode_reward)
        
        print(scores)
        print(np.mean(scores))
    
    def save(self, path, episode):
        self.actor.save_model("{}{}".format(path, episode))
        self.critic.save_model("{}{}".format(path, episode))

    def load(self, path_actor, path_critic):
        self.critic.load_model(path_critic)
        self.actor.load_model(path_actor)
    
    def save_rewards(self, episode_num, step, reward):
        with open("rewards_{}_{}.csv".format(self.replay_buffer.buffer_size, self.minibatch_size), "a") as f:
            f.write("{}, {}, {}, {}\n".format(datetime.now(), episode_num, step, reward))

