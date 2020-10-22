import gym
from gym import Wrapper
import cv2 as cv
import numpy as np
import math

from main import Side, get_animations_of_character, SpriteDetector, KenHadokenProjectileSpriteDetector, \
    get_animation_sprites
from sprite_embedding import generate_sprite_name


class SuperStreetFighter2ObservationSpaceWrapper(Wrapper):
    def __init__(self, env, animation_embeddings, sprite_embeddings):
        super().__init__(env)
        self._animation_embeddings = animation_embeddings
        self._sprite_embeddings = sprite_embeddings
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(
                2 * 3
                + self._animation_embeddings[0].size() + self._sprite_embeddings[0].size()
                + self._animation_embeddings[1].size() + self._sprite_embeddings[1].size()
                + 4,
            )
        )

        animations = (
            get_animations_of_character('ryu'),
            get_animations_of_character('ken')
        )
        self.sprite_detectors = (
            SpriteDetector(animations[0]),
            SpriteDetector(animations[1]),
        )
        self.projectile_sprite_detectors = (
            None,
            KenHadokenProjectileSpriteDetector(
                get_animation_sprites('sprites/projectiles/ken/hadoken_projectile')
            )
        )

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return (0,) * self.observation_space.shape[0]

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation, info), reward, done, info

    def observation(self, observation, info):
        observations = cv.cvtColor(observation, cv.COLOR_RGB2GRAY)
        results = [None, None]
        # print('Ryu:')
        results[0] = self.sprite_detectors[0].detect(
            observations,
            Side.LEFT if info['x_p1'] <= info['x_p2'] else Side.RIGHT
        )
        # print('Ken:')
        results[1] = self.sprite_detectors[1].detect(
            observations,
            Side.LEFT if info['x_p2'] <= info['x_p1'] else Side.RIGHT
        )
        # print(results)

        projectile_results = [None, None]
        projectile_results[0] = None
        projectile_results[1] = self.projectile_sprite_detectors[1].detect(observations)
        # print(projectile_results)

        return np.concatenate(
            (
                generate_character_observations(
                    self._animation_embeddings[0],
                    self._sprite_embeddings[0],
                    {
                        'results': results[0],
                        'hp': info['hp_p1'],
                        'x': info['x_p1'],
                        'y': info['y_p1']
                    }
                ),
                generate_character_observations(
                    self._animation_embeddings[1],
                    self._sprite_embeddings[1],
                    {
                        'results': results[1],
                        'hp': info['hp_p2'],
                        'x': info['x_p2'],
                        'y': info['y_p2']
                    }
                ),
                (
                    normalize_time_left(info['time_left']),
                    normalize_x_distance(calculate_x_distance(info['x_p1'], info['x_p2'])),
                    normalize_y_distance(calculate_y_distance(info['y_p1'], info['y_p2'])),
                    normalize_distance(calculate_distance(info['x_p1'], info['y_p1'], info['x_p2'], info['y_p2']))
                )
            )
        )


def generate_character_observations(animation_embedding, sprite_embedding, character_observations):
    return np.concatenate((
        generate_animation_observations(animation_embedding, character_observations['results']),
        generate_sprite_observations(sprite_embedding, character_observations['results']),
        (
            # normalize_animation_frame_number(character_observations['animation_frame']),
            normalize_hp(character_observations['hp']),
            normalize_x_coordinate(character_observations['x']),
            normalize_y_coordinate(character_observations['y']),
        )
    ))


def generate_animation_observations(animation_embedding, results):
    observations = [0] * animation_embedding.size()
    for result in results:
        id = animation_embedding.value_to_id(result['name'])
        index = id - 1
        observations[index] = 1
    return observations


def generate_sprite_observations(sprite_embedding, results):
    observations = [0] * sprite_embedding.size()
    for result in results:
        id = sprite_embedding.value_to_id(
            generate_sprite_name(
                result['name'],
                result['sprite_number']
            )
        )
        index = id - 1
        observations[index] = 1
    return observations


def calculate_x_distance(x1, x2):
    return abs(x2 - x1)


def calculate_y_distance(y1, y2):
    return abs(y2 - y1)


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def normalize_animation_frame_number(animation_frame_number):
    return animation_frame_number / 50.0


def normalize_hp(hp):
    return hp / 176.0


def normalize_x_coordinate(x):
    return (x - 54) / float(459 - 54)


def normalize_y_coordinate(y):
    return y / 192.0


def normalize_time_left(time_left):
    return time_left / 99.0


def normalize_x_distance(x_distance):
    return x_distance / float(459 - 54)


def normalize_y_distance(y_distance):
    return y_distance / 192.0


def normalize_distance(distance):
    distance / calculate_distance(54, 0, 459, 192)
