from experiments.model.train import create_sf2_env
from experiments.ppo2.ppo2_sf2.sprite_ids_when_able_to_input import sprite_ids_when_able_to_input

import os
import numpy as np

os.chdir('../..')

FPS = 30

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    env = create_sf2_env()

    observations = env.reset()
    idle_action = (0,) * env.action_space.n

    number_of_examples = 1000 * 1000
    X = []
    Y = []
    for iteration_number in range(number_of_examples):
        action = env.action_space.sample()
        observations_to_choose_action_based_on = observations
        observations, reward, done, info = env.step(action)
        total_reward = reward
        # env.render()

        while not done and info['sprite_id_next_frame_p1'] not in sprite_ids_when_able_to_input:
            observations, reward, done, info = env.step(idle_action)
            total_reward += reward
            # env.render()

        multi_binary_action = env.map_discrete_action_to_multi_binary_action(action)
        multi_binary_action = tuple(1 if value else 0 for value in multi_binary_action)
        X.append(np.concatenate((multi_binary_action, observations_to_choose_action_based_on)))
        Y.append(np.array((normalize_reward(total_reward),)))

        if done:
            observations = env.reset()

    X = np.array(X)
    Y = np.array(Y)

    np.savez('data', X=X, Y=Y)


def normalize_reward(reward):
    return 0.5 + (reward / 2.0)


if __name__ == '__main__':
    main()
