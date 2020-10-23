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
        state='ryu_vs_ken_long_highest_difficulty',
        scenario=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        obs_type=None,
        players=2,
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
    number_of_batches = 5

    class PlayerMove:
        def __init__(self, observation, action):
            self.observations_to_choose_action_based_on = observation
            self.action = action
            self.total_reward = 0
            self.out_of_recovery = False
            self.opponent_out_of_recovery = False

    player_moves = [None] * env.env.players

    batch_number = 0

    X = []
    Y = []

    while batch_number < number_of_batches:
        i = 0

        actions = [None, None]
        for player in range(2):
            move = player_moves[player]
            if move:
                action = idle_action
            else:
                action = env.action_space.sample()[0:env.env.action_space.n]
                move = PlayerMove(observations, action)
                player_moves[player] = move
            actions[player] = action

        observations, rewards, done, info = env.step(actions)

        for player in range(2):
            move = player_moves[player]
            move.total_reward += rewards[player]
            player_id = player + 1
            opponent_player_id = (player + 1) % 2 + 1
            move.out_of_recovery = move.out_of_recovery or info['sprite_id_next_frame_p' + str(player_id)] in sprite_ids_when_able_to_input
            move.opponent_out_of_recovery = move.opponent_out_of_recovery or info['sprite_id_next_frame_p' + str(opponent_player_id)] in sprite_ids_when_able_to_input

            if move.out_of_recovery and move.opponent_out_of_recovery:
                multi_binary_action = env.map_discrete_action_to_multi_binary_action(move.action)
                multi_binary_action = tuple(1 if value else 0 for value in multi_binary_action)
                X.append(np.concatenate((multi_binary_action, move.observations_to_choose_action_based_on)))
                Y.append(np.array((normalize_reward(move.total_reward),)))
                i += 1

                if i == batch_size:
                    X = np.array(X)
                    Y = np.array(Y)

                    model.fit(X, Y, epochs=1)

                    X = []
                    Y = []

                    batch_number += 1

        if done:
            observations = env.reset()

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
