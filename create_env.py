import gym
import numpy as np
from gym import spaces
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

def make_submarine_env(path_to_env: str, use_visual: bool = False, no_graphics: bool = True):
    unity_env = UnityEnvironment(file_name=path_to_env, no_graphics=no_graphics)
    env = UnityToGymWrapper(unity_env, uint8_visual=use_visual)
    return env

env = make_submarine_env("D:\Engineering\AUVT\unity_export\0008-AUVSim_With_Python_Interface.exe")

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

env.close()
