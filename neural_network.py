import gym
from snake_rl.envs import SnakeEnv, Actions

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory


def create_neural_network():
    env = gym.make('snake-v1', window_size=(1280, 720))
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Flatten(input_shape=env.observation_space.shape))
    model.add(Dense(18))
    model.add(Activation('relu'))
    model.add(Dense(18))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    memory = SequentialMemory(limit=20000, window_length=1)
    policy = EpsGreedyQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10000,
                   target_model_update=0.01, policy=policy, batch_size=64)
    dqn.compile(optimizer=Adam(lr=0.001), metrics=['mae'])

    return env, dqn


if __name__ == '__main__':
    env, dqn = create_neural_network()
    env.fps = 99999
    dqn.fit(env, nb_steps=1000000, visualize=True, verbose=0)
    dqn.save_weights(f'dqn_snake_weights.h5f', overwrite=True)
    input('Finalizado')
    env.fps = 10
    dqn.test(env, nb_episodes=5, visualize=True)
