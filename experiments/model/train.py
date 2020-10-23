from experiments.ppo2.ppo2_sf2.sprite_ids_when_able_to_input import sprite_ids_when_able_to_input

import retro
import os
import tensorflow as tf
import numpy as np

from experiments.ppo2.RoundResultOutputWrapper import RoundResultOutputWrapper
from experiments.ppo2.RyuDiscretizer import RyuDiscretizer

from experiments.ppo2.SuperStreetFigher2ObservationSpaceWrapper import SuperStreetFighter2ObservationSpaceWrapper

os.chdir('../..')

FPS = 30

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model')


def create_sf2_env(print_round_results=False):
    retro.data.Integrations.add_custom_path(
        '/home/jonas/Documents'
    )
    env = retro.make(
        game='SuperStreetFighter2-Snes',
        state='ryu_vs_fei_long_highest_difficulty',
        scenario=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        obs_type=None,
        players=1,
        use_restricted_actions=retro.Actions.FILTERED,
    )
    env = SuperStreetFighter2ObservationSpaceWrapper(env)
    # env = MonitorWrapper(env)
    if print_round_results:
        env = RoundResultOutputWrapper(env)
    env = RyuDiscretizer(env)
    return env


def main():
    env = create_sf2_env()
    model = create_model(env)

    observations = env.reset()
    idle_action = (0,) * env.action_space.n

    batch_size = 1000
    number_of_iterations = 5
    for iteration_number in range(number_of_iterations):
        X = []
        Y = []

        for i in range(batch_size):
            action = env.action_space.sample()
            observations_to_choose_action_based_on = observations
            observations, reward, done, info = env.step(action)
            total_reward = reward
            # env.render()

            out_of_recovery = [
                info['sprite_id_next_frame_p1'] in sprite_ids_when_able_to_input,
                info['sprite_id_next_frame_p2'] in sprite_ids_when_able_to_input
            ]
            while not done and (not out_of_recovery[0] or not out_of_recovery[1]):
                observations, reward, done, info = env.step(idle_action)
                total_reward += reward
                # env.render()
                out_of_recovery[0] = out_of_recovery[0] or info['sprite_id_next_frame_p1']
                out_of_recovery[1] = out_of_recovery[1] or info['sprite_id_next_frame_p2']

            multi_binary_action = env.map_discrete_action_to_multi_binary_action(action)
            multi_binary_action = tuple(1 if value else 0 for value in multi_binary_action)
            X.append(np.concatenate((multi_binary_action, observations_to_choose_action_based_on)))
            Y.append(np.array((normalize_reward(total_reward),)))

            if done:
                observations = env.reset()

        X = np.array(X)
        Y = np.array(Y)

        model.fit(X, Y, epochs=1)

    model.save(MODEL_PATH)


def normalize_reward(reward):
    return 0.5 + (reward / 2.0)


def create_model(env):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(env.env.action_space.n + env.observation_space.shape[0], activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    main()
