import time

from tensorflow import keras
from experiments.model.train import create_model, create_sf2_env
from experiments.ppo2.ppo2_sf2.sprite_ids_when_able_to_input import sprite_ids_when_able_to_input

import os
import numpy as np

os.chdir('../..')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model')


FPS = 30
SPEED = 1
duration_per_frame = 1 / (SPEED * FPS)  # in seconds


def main():
    env = create_sf2_env(print_round_results=True)
    model = keras.models.load_model(MODEL_PATH)

    observations = env.reset()
    idle_action = (0,) * env.action_space.n

    while True:
        frame_start_time = time.time()

        X = []
        for action in range(env.action_space.n):
            multi_binary_action = env.map_discrete_action_to_multi_binary_action(action)
            multi_binary_action = tuple(1 if value else 0 for value in multi_binary_action)
            X.append(np.concatenate((multi_binary_action, observations)))

        X = np.array(X)
        evaluations = model.predict(X)
        action, _ = max(
            enumerate(evaluations),
            key=lambda evaluation: evaluation[1][0]
        )

        observations, reward, done, info = env.step(action)
        env.render()

        frame_end_time = time.time()
        frame_duration = frame_end_time - frame_start_time
        frame_sleep_duration = duration_per_frame - frame_duration
        if frame_sleep_duration > 0:
            time.sleep(frame_sleep_duration)

        while not done and info['sprite_id_next_frame_p1'] not in sprite_ids_when_able_to_input:
            frame_start_time = time.time()

            observations, reward, done, info = env.step(idle_action)
            env.render()

            frame_end_time = time.time()
            frame_duration = frame_end_time - frame_start_time
            frame_sleep_duration = duration_per_frame - frame_duration
            if frame_sleep_duration > 0:
                time.sleep(frame_sleep_duration)

        if done:
            observations = env.reset()


if __name__ == '__main__':
    main()
