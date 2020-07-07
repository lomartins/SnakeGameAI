from gym.envs.registration import register

register(
    id='snake-v1',
    entry_point='snake_rl.envs:SnakeEnv'
)
