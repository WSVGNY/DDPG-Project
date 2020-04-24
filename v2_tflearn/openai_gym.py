import gym

class LunarLanderGym:
    def __init__(self):
        self.env = gym.make('LunarLanderContinuous-v2')

    def reset(self):
        self.env.reset()
        
    def close(self):
        self.env.close()

