import numpy as np
from actor import Actor
from critic import Critic
from utils.actionnoise import OrnsteinUhlenbeckProcess
from utils.replaybuffer import ReplayBuffer
import tensorflow as tf

MINIBATCH_SIZE = 64

class Agent:
    def __init__(self, states_dim, actions_dim, buffer_size = 20000, gamma = 0.99, lr = 0.00005, tau = 0.001):
        self.states_dim = states_dim
        self.actions_dim = actions_dim
        self.gamma = gamma
        self.lr = lr
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.actor = Actor(self.states_dim, self.actions_dim, 0.1 * lr, tau)
        self.critic = Critic(self.states_dim, self.actions_dim, lr, tau)

    def bellman(self, rewards, dones, next_state_rewards):
        target_rewards = np.asarray(next_state_rewards)
        for i in range(MINIBATCH_SIZE):
            if dones[i]:
                target_rewards[i] = rewards[i] 
            else:
                target_rewards[i] = rewards[i] + self.gamma * next_state_rewards[i]
        
        return target_rewards

    def train(self, env, nb_episodes=5000, render=False):
        for i in range(nb_episodes):
            
            # Reinitialiser le jeu
            state = env.reset()
            episode_reward = 0
            done = False
            noise = OrnsteinUhlenbeckProcess(size = self.actions_dim)
            time = 0

            # For plotting
            episodes_rewards = []

            while not done:
                if render: env.render()
                action = np.clip(self.actor.choose_action(np.expand_dims(state, 0))[0] + noise.generate(time), -1, 1) 
                next_state, reward, done, info = env.step(action)
                self.replay_buffer.store(state, action, reward, done, next_state)
                
                state = next_state
                episode_reward += reward
                time += 1
        
                if self.replay_buffer.get_size() > MINIBATCH_SIZE:
                    # Je fais un minibatch mother-fucker
                    states, actions, rewards, dones, next_states = self.replay_buffer.sample(MINIBATCH_SIZE)
                    #q_next
                    next_state_rewards = self.critic.evaluate_action_target(next_states, self.actor.choose_action_target(next_states))
                    target_rewards = self.bellman(rewards, dones, next_state_rewards)

                    # Update models
                    self.critic.train(states, actions, target_rewards)
                    # training_actions = self.actor.choose_action(states)
                    # training_actions_gradients = self.critic.get_action_gradients(states, training_actions)
                    
                    with tf.GradientTape() as tape:
                        y_pred = self.actor.model(states)
                        q_pred = self.critic.model([states, y_pred])
                    critic_grads = tape.gradient(q_pred, y_pred)
                    
                    # self.actor.train(states, training_actions_gradients)
                    self.actor.train(states, critic_grads)

                    # Update target models
                    self.actor.update_target_model()
                    self.critic.update_target_model()
            
            print(episode_reward)
            episodes_rewards.append(episode_reward)
            self.save("./saved_models/", i)
            self.save_rewards(i, episode_reward)

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
        path += '_LR_{}_ep{}'.format(self.lr, episode)
        self.actor.save_model(path)
        self.critic.save_model(path)

    def load(self, path_actor, path_critic):
        self.critic.load_model(path_critic)
        self.actor.load_model(path_actor)
    
    def save_rewards(self, episode_num, reward):
        with open("rewards.csv", "a") as f:
            f.write("{}, {}".format(episode_num, reward))

