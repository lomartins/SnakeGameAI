from train_snake import create_neural_network


env, dqn = create_neural_network()
dqn.load_weights('dqn_snake_weights.h5f')
dqn.test(env, nb_episodes=1, visualize=True)
