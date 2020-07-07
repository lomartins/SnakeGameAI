from abc import ABC
import gym
from gym import spaces
import numpy as np
from gym.utils import seeding
from enum import Enum
import random
import pygame
import pygame.locals
import math
import time

snake_image = pygame.image.load('snake.jpg')
snake_image = pygame.transform.scale(snake_image, (320, 280))


def on_grid_random(max_xy):
    x = random.randint(10, max_xy[0] - 10)
    y = random.randint(10, max_xy[1] - 10)
    return x // 10 * 10, y // 10 * 10


def align_on_grid(vector):
    return int(vector // 10 * 10)


def collision(c1, c2):
    return c1 == c2


def get_distance(pos1, pos2):
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** (1 / 2)


class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Coords:
    def __init__(self, xy):
        self.x = xy[0]
        self.y = xy[1]
        self.xy = xy


class SnakeEnv(gym.Env, ABC):

    def __init__(self, window_size=(1280, 720)):
        self.window_size = window_size
        self.seed()
        self.shape = (1, 16)
        self.fps = 600

        wall = []
        max_grid = align_on_grid((window_size[0] * 6 / 10)), window_size[1]
        self.max_grid = max_grid
        for i in range(0, max_grid[0]):
            wall.append((i, 0))
            wall.append((i, window_size[1] - 10))
        for i in range(0, window_size[1], 10):
            wall.append((0, i))
            wall.append((max_grid[0], i))

        self.wall = wall
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=self.shape, dtype=np.float32)
        self._first_render = True
        self.deaths = 0
        self.count_step = 0

        self.observation = None
        self.screen = None
        self._done = None
        self._game_over = None
        self._total_reward = None
        self.snake = None
        self.my_direction = None
        self.apple_pos = None
        self.distance = None
        self.previous_distance = None
        self._hunger_steps = None

        self.wall_sprite = pygame.Surface((10, 10))
        self.wall_sprite.fill((100, 100, 100))

        self.snake_skin = pygame.Surface((10, 10))
        self.snake_skin.fill((255, 255, 255))

        self.apple = pygame.Surface((10, 10))
        self.apple.fill((255, 0, 0))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def rand_apple(self):
        self.apple_pos = on_grid_random(self.max_grid)
        return self.apple_pos

    def reset(self):
        self._done = False
        self._hunger_steps = 0
        self._game_over = False
        midx = self.window_size[0] // 10 * 5
        midy = self.window_size[1] // 10 * 5
        self.snake = [(midx, midy), (midx + 10, midy), (midx + 20, midy)]
        self.my_direction = Actions.LEFT
        self.rand_apple()
        self.distance = get_distance(self.apple_pos, self.snake[0])
        self.previous_distance = self.distance
        self._total_reward = 0
        return self._get_observation()

    def step(self, action):
        self.count_step += 1
        self._done = False
        self._game_over = False
        self._hunger_steps += 1
        self.distance = get_distance(self.snake[0], self.apple_pos)

        self.my_direction = Actions(action)

        if self.my_direction == Actions.UP:
            self.snake[0] = (self.snake[0][0], self.snake[0][1] - 10)
        if self.my_direction == Actions.DOWN:
            self.snake[0] = (self.snake[0][0], self.snake[0][1] + 10)
        if self.my_direction == Actions.RIGHT:
            self.snake[0] = (self.snake[0][0] + 10, self.snake[0][1])
        if self.my_direction == Actions.LEFT:
            self.snake[0] = (self.snake[0][0] - 10, self.snake[0][1])

        if collision(self.snake[0], self.apple_pos):
            self.rand_apple()
            self.snake.append((0, 0))
            self._hunger_steps = 0

        for i in range(len(self.snake) - 1, 0, -1):
            if collision(self.snake[0], self.snake[i]):
                self._game_over = True
            self.snake[i] = (self.snake[i - 1][0], self.snake[i - 1][1])

        if self.snake[0] in self.wall:
            self._game_over = True

        step_reward = self._calculate_reward()
        self._total_reward += step_reward

        self.previous_distance = self.distance

        observation = self._get_observation()
        self.observation = observation
        if self._game_over:
            self.deaths += 1

        if self._game_over or self._hunger_steps > 300:
            self._done = True

        info = dict(
            total_reward=self._total_reward,
            direction=self.my_direction.value
        )
        return observation, step_reward, self._done, info

    def _get_observation(self):
        head = self.snake[0]

        def sensor(x_increment, y_increment, head_pos):
            sensor_pos = (head_pos[0] + x_increment, head_pos[1] + y_increment)
            if sensor_pos in self.wall or sensor_pos in self.snake[1:]:
                return 1

            return 0

        indexes = [0, 10, -10]

        apple_detector = []
        apple_detected = -1
        min_distance = self.window_size[0] + self.window_size[1]
        count = 0
        for i in indexes:
            for j in indexes:
                if i == j == 0:
                    continue
                pos = (head[0] + i, head[1] + j)
                distance = get_distance(pos, self.apple_pos)
                if distance < min_distance:
                    min_distance = distance
                    apple_detected = count
                count += 1

        for i in range(8):
            if i == apple_detected:
                apple_detector.append(1)
            else:
                apple_detector.append(0)

        observation = np.array([
            # up
            sensor(0, -10, head),

            # up right
            sensor(10, -10, head),

            # right
            sensor(10, 0, head),

            # down right
            sensor(10, 10, head),

            # down
            sensor(0, 10, head),

            # down left
            sensor(-10, 10, head),

            # left
            sensor(-10, 0, head),

            # up left
            sensor(-10, -10, head),

            apple_detector[0],

            apple_detector[1],

            apple_detector[2],

            apple_detector[3],

            apple_detector[4],

            apple_detector[5],

            apple_detector[6],

            apple_detector[7],

        ])
        self.observation = observation
        return observation

    def _calculate_reward(self):
        reward = 0
        if self.previous_distance > self.distance:
            reward += 5
        else:
            reward -= 7

        if collision(self.snake[0], self.apple_pos):
            reward += 30

        if self._game_over:
            reward -= 100

        return reward

    def start_game(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size[0], self.window_size[1]))
        pygame.display.set_caption('Snake')

    def render(self, mode='human', screen=None, first=True, human=False):
        if self._first_render and first:
            self.start_game()
            self._first_render = False

        if screen is None:
            screen = self.screen

        if mode == 'human':
            pygame.time.Clock().tick(self.fps)
            if not human:
                for event in pygame.event.get():
                    if event.type == pygame.locals.QUIT:
                        pygame.quit()
                        exit()

                    if event.type == pygame.locals.KEYDOWN:
                        if event.key == pygame.locals.K_UP:
                            self.fps += 10
                        if event.key == pygame.locals.K_DOWN:
                            self.fps -= 10
                        if event.key == pygame.locals.K_LEFT:
                            self.fps = 15
                        if event.key == pygame.locals.K_RIGHT:
                            self.fps = 99999999

            screen.fill((0, 0, 0))
            for wall_pos in self.wall:
                screen.blit(self.wall_sprite, wall_pos)

            screen.blit(self.apple, self.apple_pos)
            for pos in self.snake:
                screen.blit(self.snake_skin, pos)

            center_div_x = align_on_grid((self.window_size[0] + self.max_grid[0]) / 2)
            title_y = 10
            death_y = self.window_size[1] - 60
            button_y = self.window_size[1] - 170
            snake_image_y = self.window_size[1] - 600

            self.blit_txt('Deep Snake', (center_div_x - 100, title_y), screen)
            self.blit_txt(f'FPS: {self.fps}', (center_div_x + 150, title_y), screen)
            self.blit_txt(f'Total Reward: {self._total_reward}', (center_div_x - 100, death_y), screen)
            self.blit_txt(f'Step: {self.count_step}', (center_div_x + 150, death_y), screen)

            button = pygame.Surface((40, 40))
            button.fill((150, 150, 150))
            pressed_button = pygame.Surface((40, 40))
            pressed_button.fill((255, 0, 0))

            buttons = []
            for act in Actions:
                if act == self.my_direction:
                    buttons.append(pressed_button)
                else:
                    buttons.append(button)

            self.blit_txt('Controllers', (center_div_x - 110, button_y - 100), screen)
            screen.blit(buttons[0], (center_div_x - 130, button_y - 40))
            screen.blit(buttons[1], (center_div_x + 40 - 130, button_y))
            screen.blit(buttons[2], (center_div_x - 130, button_y + 40))
            screen.blit(buttons[3], (center_div_x - 40 - 130, button_y))

            apple_sensor = self.observation[-8:]

            self.blit_txt('Apple Sensors', (center_div_x + 120, button_y - 100), screen)
            indexes = [0, 40, -40]
            count = 0
            for i in indexes:
                for j in indexes:
                    if i == j == 0:
                        continue
                    if apple_sensor[count] == 1:
                        sprite = pressed_button
                    else:
                        sprite = button
                    screen.blit(sprite, (center_div_x + 100 + i, button_y + j))
                    count += 1
            screen.blit(snake_image, (center_div_x - 130, snake_image_y))
            pygame.display.update()

    def blit_txt(self, txt, pos, screen):
        font_name = pygame.font.match_font('arial')
        font = pygame.font.Font(font_name, int(self.window_size[0] * 3 / 100))
        text_surface = font.render(txt, True, (255, 255, 255))
        text_rect = text_surface.get_rect()
        text_rect.midtop = (pos[0], pos[1])
        screen.blit(text_surface, text_rect)

    def is_done(self):
        return self._done

    def human_render(self):
        action = self.my_direction.value
        self.render(human=True)
        for event in pygame.event.get():
            if event.type == pygame.locals.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.locals.KEYDOWN:
                if event.key == pygame.locals.K_UP and self.my_direction != Actions.DOWN:
                    action = Actions.UP
                if event.key == pygame.locals.K_DOWN and self.my_direction != Actions.UP:
                    action = Actions.DOWN
                if event.key == pygame.locals.K_LEFT and self.my_direction != Actions.RIGHT:
                    action = Actions.LEFT
                if event.key == pygame.locals.K_RIGHT and self.my_direction != Actions.LEFT:
                    action = Actions.RIGHT
        return action
