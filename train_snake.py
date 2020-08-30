import gym
from snake_rl.envs import SnakeEnv, Actions

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

def create_neural_network(enviroment):

    nb_actions = enviroment.action_space.n

    model = Sequential()
    model.add(Flatten(input_shape=enviroment.observation_space.shape))
    model.add(Dense(18))
    model.add(Activation('relu'))
    model.add(Dense(18))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    memory = SequentialMemory(limit=20000, window_length=1)
    policy = EpsGreedyQPolicy(0.1)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
                   target_model_update=0.01, policy=policy, batch_size=64)
    dqn.compile(optimizer=Adam(lr=0.001), metrics=['mae'], )

    return dqn


if __name__ == '__main__':
    env = gym.make('snake-v1', window_size=(1280, 720))
    dqn = create_neural_network(env)

    env.fps = 1
    dqn.fit(env, nb_steps=1000000, callbacks=[ModelIntervalCheckpoint('dqn_snake_weights-bkp.h5f', interval=10000)], visualize=True, verbose=0)
    dqn.save_weights(f'dqn_snake_weights.h5f', overwrite=True)
    input('Done')
    env.fps = 10
    dqn.test(env, nb_episodes=5, visualize=True)
