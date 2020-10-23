import retro
import os
import tensorflow as tf
from baselines.common.vec_env import SubprocVecEnv, DummyVecEnv

from experiments.ppo2.MonitorWrapper import MonitorWrapper
from experiments.ppo2.ppo2_sf2 import ppo2

from experiments.ppo2.RoundResultOutputWrapper import RoundResultOutputWrapper
from experiments.ppo2.RyuDiscretizer import RyuDiscretizer, RyuDiscretizerDefending

from experiments.ppo2.SuperStreetFigher2ObservationSpaceWrapper import SuperStreetFighter2ObservationSpaceWrapper

os.chdir('../..')

FPS = 30

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(SCRIPT_DIR, 'model')
CHECKPOINTS_PATH = os.path.join(LOG_PATH, 'checkpoints')
MODEL_PATH = os.path.join(CHECKPOINTS_PATH, 'latest')


def create_sf2_env():
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
    env = RoundResultOutputWrapper(env)
    env = RyuDiscretizer(env)
    return env


def main():
    os.environ['OPENAI_LOGDIR'] = LOG_PATH

    number_of_environments = 1
    venv = DummyVecEnv([create_sf2_env] * number_of_environments)

    ppo2.learn(
        network='mlp',
        env=venv,
        # eval_env=venv,
        total_timesteps=40000000,
        nminibatches=number_of_environments,
        lam=0.95,
        gamma=0.99,
        noptepochs=3,
        log_interval=1000,
        ent_coef=.01,
        lr=lambda alpha: 2.5e-4 * alpha,
        vf_coef=1.0,
        cliprange=lambda alpha: 0.1 * alpha,
        save_interval=1000,
        # load_path=MODEL_PATH,
        # neuronal network parameters
        activation=tf.nn.relu,
        num_layers=4,  # 4, 2
        num_hidden=128,  # 64, 64
        layer_norm=True
    )


if __name__ == '__main__':
    main()
