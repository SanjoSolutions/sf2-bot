import os
import time

import tensorflow as tf
from baselines.common.vec_env import DummyVecEnv

from experiments.ppo2.learn import create_sf2_env
from experiments.ppo2.ppo2_sf2 import ppo2
from experiments.ppo2.ppo2_sf2.sprite_ids_when_able_to_input import sprite_ids_when_able_to_input

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(SCRIPT_DIR, 'model')
CHECKPOINTS_PATH = os.path.join(LOG_PATH, 'checkpoints')
MODEL_PATH = os.path.join(CHECKPOINTS_PATH, '3661000')


os.chdir('../..')


FPS = 30
SPEED = 1
duration_per_frame = 1 / (SPEED * FPS)  # in seconds


def main():
    number_of_environments = 1
    venv = DummyVecEnv([create_sf2_env] * number_of_environments)

    model = ppo2.predict(
        network='mlp',
        env=venv,
        nsteps=5 * FPS,
        nminibatches=number_of_environments,
        vf_coef=1.0,
        load_path=MODEL_PATH,
        # neuronal network parameters
        activation=tf.nn.relu,
        num_layers=4,  # 4, 2
        num_hidden=128,  # 64, 64
        layer_norm=False
    )

    observations = venv.reset()
    info = None
    idle_action = (0,) * venv.envs[0].action_space.n

    while True:
        frame_start_time = time.time()
        if info is not None and info[0]['sprite_id_next_frame_p1'] in sprite_ids_when_able_to_input:
            actions, _, _, _ = model.step(observations)
        else:
            actions = idle_action

        observations, rewards, done, info = venv.step(actions)
        venv.render()

        frame_end_time = time.time()
        frame_duration = frame_end_time - frame_start_time
        frame_sleep_duration = duration_per_frame - frame_duration
        if frame_sleep_duration > 0:
            time.sleep(frame_sleep_duration)

        if done:
            observations = venv.reset()


if __name__ == "__main__":
    main()
