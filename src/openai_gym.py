import gym

class LunarLanderGym:
    def __init__(self):
        self.env = gym.make('LunarLanderContinuous-v2')

    def play(self):
        self.env.reset()
        done = False
        while not done:
            self.env.render()
            observation, reward, done, info = self.env.step([0,0,-1])
            print(reward)
            print(observation)
    def reset(self):
        self.env.reset()
        
    def close(self):
        self.env.close()

