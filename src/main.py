from openai_gym import LunarLanderGym

gym = LunarLanderGym()

for i in range(1000):
    gym.play()