import retro
import os
import tensorflow as tf
from baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder

from animation_embedding import create_animation_embedding
from experiments.ppo2.ppo2 import ppo2
from experiments.ppo2.RyuDiscretizer import RyuDiscretizer, RyuDiscretizerDefending

from experiments.ppo2.SuperStreetFigher2ObservationSpaceWrapper import SuperStreetFighter2ObservationSpaceWrapper
from main import get_animations_of_character
from sprite_embedding import create_sprite_embedding

os.chdir('../..')

FPS = 30

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(SCRIPT_DIR, 'model')
CHECKPOINTS_PATH = os.path.join(LOG_PATH, 'checkpoints')
MODEL_PATH = os.path.join(CHECKPOINTS_PATH, 'latest')


def make_sf2_env(animation_embeddings, sprite_embeddings):
    retro.data.Integrations.add_custom_path(
        '/Applications'
    )
    env = retro.make(
        game='SuperStreetFighter2-Snes',
        state='ryu_vs_ken_both_controlled',
        scenario=None,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        obs_type=retro.Observations.IMAGE,
        players=1,
        use_restricted_actions=retro.Actions.FILTERED,
    )
    env = SuperStreetFighter2ObservationSpaceWrapper(env, animation_embeddings, sprite_embeddings)
    env = RyuDiscretizer(env)
    return env


def main():
    os.environ['OPENAI_LOGDIR'] = LOG_PATH

    number_of_environments = 1
    animations = (
        get_animations_of_character('ryu'),
        get_animations_of_character('ken')
    )
    animation_embeddings = tuple(
        create_animation_embedding(animations)
        for animations
        in animations
    )
    sprite_embeddings = tuple(
        create_sprite_embedding(animations)
        for animations
        in animations
    )
    create_sf2_env = lambda: make_sf2_env(animation_embeddings, sprite_embeddings)
    venv = DummyVecEnv([create_sf2_env] * number_of_environments)
    video_path = './recording'
    video_length = 10 * FPS
    venv = VecVideoRecorder(venv, video_path, record_video_trigger=lambda step: step %
                            video_length == 0, video_length=video_length)
    ppo2.learn(
        network='mlp',
        env=venv,
        # eval_env=venv,
        total_timesteps=40000000,
        nsteps=1,
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
        num_layers=2,  # 4, 2
        num_hidden=48,  # 64, 64
        layer_norm=False
    )


if __name__ == '__main__':
    main()