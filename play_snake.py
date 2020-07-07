import gym
import pygame
from pygame.locals import *
from snake_rl.envs import SnakeEnv, Actions

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

env = gym.make('snake-v1', window_size=(1280, 720))
env.fps = 20
env.reset()
action = LEFT

while True:
    if env.is_done():
        env.reset()
        action = env.my_direction.value
    action = env.human_render()
    env.step(action)
