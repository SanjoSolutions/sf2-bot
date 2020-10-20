from enum import IntEnum

import cv2 as cv
import numpy as np
import os
from os import path
import itertools


class Side(IntEnum):
    LEFT = 0
    RIGHT = 1


def get_images_in_folder(folder_path):
    return read_images(get_image_paths_for_images_in_folder(folder_path))


def get_image_paths_for_images_in_folder(folder_path):
    return file_names_to_absolute_paths(folder_path, list_dir(folder_path))


def file_names_to_absolute_paths(base_path, file_names):
    return tuple(file_name_to_absolute_path(base_path, file_name) for file_name in file_names)


def file_name_to_absolute_path(base_path, file_name):
    return path.join(base_path, file_name)


def get_ordered_numbered_images_in_folder(folder_path):
    return read_images(get_ordered_numbered_file_paths(folder_path))


def get_ordered_numbered_file_paths(folder_path):
    return file_names_to_absolute_paths(folder_path, get_ordered_numbered_file_names(folder_path))


def get_ordered_numbered_file_names(folder_path):
    return sorted(list_dir(folder_path), key=file_name_to_integer)


def file_name_to_integer(file_name):
    return int(path.splitext(file_name)[0])


def list_dir(folder_path):
    return tuple(
        file_name
        for file_name in os.listdir(folder_path)
        if not file_name.startswith('.')
    )


def read_images(image_paths):
    return tuple(read_image(image_path) for image_path in image_paths)


def read_image(image_path):
    return cv.imread(image_path, flags=cv.IMREAD_GRAYSCALE)


def get_animations_of_character(character_name):
    animations = {}
    for animation_folder in get_animation_folders_for_character(character_name):
        animation_name = path.basename(animation_folder)
        animation_sprite_paths = get_ordered_numbered_file_paths(animation_folder)
        animation_sprites = []

        for sprite_path in animation_sprite_paths:
            sprite_number = get_sprite_number_from_path(sprite_path)
            image = cv.imread(sprite_path, flags=cv.IMREAD_UNCHANGED)
            channels = cv.split(image)
            alpha_channel = np.array(channels[3]) \
                if len(channels) >= 4 \
                else np.full_like(channels[0], 255)
            mask = alpha_channel
            image_gray = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)

            image_gray_mirrored = cv.flip(image_gray, 1)
            mask_mirrored = cv.flip(mask, 1)

            animation_sprites.append((
                animation_name,
                sprite_number,
                sprite_path,
                image_gray,
                mask,
                image_gray_mirrored,
                mask_mirrored
            ))

        animations[animation_name] = animation_sprites

    return animations


def get_sprite_number_from_path(sprite_path):
    return file_name_to_integer(path.basename(sprite_path))


def get_animation_folders_for_character(character_name):
    base_path = 'sprites'
    character_sprites_path = path.join(base_path, character_name)
    animation_folders = tuple(
        folder_path
        for folder_path
        in tuple(
            path.join(character_sprites_path, animation_name)
            for animation_name
            in os.listdir(character_sprites_path)
        )
        if path.isdir(folder_path)
    )
    return animation_folders


def get_sprites_from_animations(animations):
    return tuple(itertools.chain.from_iterable(animations.values()))


def detect_sprite(sprites, side, image_to_find_in):
    method = cv.TM_CCOEFF_NORMED
    results = []
    for sprite in sprites:
        animation_name, sprite_number, sprite_path, image_left, mask_left, \
        image_right, mask_right = sprite
        if side == Side.LEFT:
            image = image_left
            mask = mask_left
        elif side == Side.RIGHT:
            image = image_right
            mask = mask_right
        result = cv.matchTemplate(image_to_find_in, image, method, mask=mask)
        _, max_value, _, max_location = cv.minMaxLoc(result)
        # print('  ', animation_name, sprite_number, max_value)
        if max_value >= 0.7:
            results.append((sprite, max_location))
    return results


def get_first_animation_frames(animations):
    first_animation_frames = [
        animation[0]
        for animation
        in animations.values()
    ]
    first_animation_frames.append(animations['knockdown_recover'][2])  # when thrown
    return first_animation_frames


class SpriteDetector:
    def __init__(self, animations):
        self.animations = animations
        self.last_possible_animations = []

    def detect(self, frame, side):
        if len(self.last_possible_animations) == 0:
            sprites = get_first_animation_frames(self.animations)
            results = detect_sprite(sprites, side, frame)
            self.last_possible_animations = []
            for sprite, location in results:
                animation_name, sprite_number, _, _, _, _, _ = sprite
                self.last_possible_animations.append({
                    'name': animation_name,
                    'frame': 1,
                    'sprite_number': sprite_number
                })
        else:
            sprites = []
            possible_last_sprite_of_an_animation = False
            for last_possible_animation in self.last_possible_animations:
                animation_name = last_possible_animation['name']
                sprite_number = last_possible_animation['sprite_number']
                sprites.extend(self.animations[animation_name][sprite_number:(sprite_number + 2)])
                if animation_name == 'tatsumaki_senpuu_kyaku' and sprite_number == 5:
                    sprites.append(self.animations[animation_name][2])
                if (
                    animation_name == 'idle' or
                    sprite_number == len(self.animations[animation_name]) - 1
                ):
                    possible_last_sprite_of_an_animation = True

            if possible_last_sprite_of_an_animation:
                sprites.extend(get_first_animation_frames(self.animations))

            results = detect_sprite(sprites, side, frame)

            previous_last_possible_animations_dictionary = dict(
                (animation['name'], animation)
                for animation
                in self.last_possible_animations
            )
            self.last_possible_animations = []
            for sprite, location in results:
                animation_name, sprite_number, _, _, _, _, _ = sprite
                previous_last_possible_animation = previous_last_possible_animations_dictionary[animation_name] \
                    if animation_name in previous_last_possible_animations_dictionary \
                    else None
                self.last_possible_animations.append({
                    'name': animation_name,
                    'frame': previous_last_possible_animation['frame'] + 1 if previous_last_possible_animation else 1,
                    'sprite_number': sprite_number
                })

        return self.last_possible_animations


def is_animation_possible(animations, animation_name):
    for animation in animations:
        if animation['name'] == animation_name:
            return True
    return False


def is_any_of_animations_possible(animations, animation_names):
    return any(
        is_animation_possible(animations, animation_name)
        for animation_name
        in animation_names
    )


def is_animation_possible_at_sprite_number_or_above(animations, animation_name, minimum_sprite_number):
    for animation in animations:
        if animation['name'] == animation_name and animation['sprite_number'] >= minimum_sprite_number:
            return True
    return False


class ActionChooser:
    def __init__(self, player_index_to_choose_for):
        self.player_index_to_choose_for = player_index_to_choose_for
        self.buffered_actions = []

    def _buffered_action(self, actions):
        self.buffered_actions.extend(actions[1:])
        return actions[0]

    # ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
    def choose_action(self, info, animations):
        pass


class ActionChooser1(ActionChooser):
    def choose_action(self, info, animations):
        if len(self.buffered_actions) >= 1:
            action = self.buffered_actions.pop(0)
            return action

        other_player_index = (self.player_index_to_choose_for + 1) % 2

        if info is None:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        players = (
            {
                'x': info['x_p1']
            },
            {
                'x': info['x_p2']
            }
        )
        distance = abs(players[0]['x'] - players[1]['x'])
        # player = players[player_index_to_choose_for]
        # opponent_index = (player_index_to_choose_for + 1) % 2
        # opponent = players[opponent_index]

        left = 0
        right = 0

        if (
            self.player_index_to_choose_for == 0 and players[0]['x'] <= players[1]['x'] or
            self.player_index_to_choose_for == 1 and players[1]['x'] <= players[0]['x']
        ):
            left = 1

        if (
            self.player_index_to_choose_for == 0 and players[0]['x'] > players[1]['x'] or
            self.player_index_to_choose_for == 1 and players[1]['x'] > players[0]['x']
        ):
            right = 1

        # FIXME: Only when character to choose for is in blocking stance
        # if distance >= 126:
        #     # hadoken
        #     return self._buffered_action((
        #         (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        #         (0, 0, 0, 0, 0, 1, (left + 1) % 2, (right + 1) % 2, 0, 0, 0, 0),
        #         (0, 0, 0, 0, 0, 0, (left + 1) % 2, (right + 1) % 2, 0, 0, 0, 0),
        #         (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        #     ))

        if (
            distance <= 47 and
            is_any_of_animations_possible(
                animations[other_player_index],
                (
                    'idle',
                    'walking',
                    'jump',
                    'forward_jump',
                    'crouch',
                    'standing_block',
                    'stunned'
                )
            )
        ):
            # throw
            return 0, 0, 0, 0, 0, 0, (left + 1) % 2, (right + 1) % 2, 0, 1, 0, 0

        if (
            distance <= 67 and
            is_animation_possible_at_sprite_number_or_above(animations[other_player_index], 'crouching_hard_kick', 2)
        ):
            # crouching hard kick
            return 0, 0, 0, 0, 0, 1, left, right, 0, 0, 0, 1

        if (
            is_any_of_animations_possible(
                animations[other_player_index],
                (
                    'jump',
                    'jumping_hard_kick',
                    'jumping_light_medium_hard_punch',
                    'jumping_light_medium_kick',
                    'forward_jump',
                    'forward_jumping_light_punch',
                    'forward_jumping_medium_hard_kick',
                    'forward_hard_kick'
                )
            )
        ):
            if distance <= 47:
                # shoryuken
                return self._buffered_action((
                    (0, 0, 0, 0, 0, 0, (left + 1) % 2, (right + 1) % 2, 0, 0, 0, 0),
                    (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
                    (0, 0, 0, 0, 0, 1, (left + 1) % 2, (right + 1) % 2, 0, 0, 0, 0),
                    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                ))

        # blocking
        return 0, 0, 0, 0, 0, 1, left, right, 0, 0, 0, 0


class ActionChooser1BestResponse(ActionChooser):
    def __init__(self, player_index_to_choose_for):
        super().__init__(player_index_to_choose_for)
        self.frame = 0
        self.next_frame_to_act = None

    def choose_action(self, info, animations):
        self.frame += 1

        if len(self.buffered_actions) >= 1:
            action = self.buffered_actions.pop(0)
            return action

        if self.next_frame_to_act is not None:
            if self.frame >= self.next_frame_to_act:
                self.next_frame_to_act = None
            else:
                return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        if info is None:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        players = (
            {
                'x': info['x_p1']
            },
            {
                'x': info['x_p2']
            }
        )

        left = 0
        right = 0

        if (
            self.player_index_to_choose_for == 0 and players[0]['x'] <= players[1]['x'] or
            self.player_index_to_choose_for == 1 and players[1]['x'] <= players[0]['x']
        ):
            left = 1

        if (
            self.player_index_to_choose_for == 0 and players[0]['x'] > players[1]['x'] or
            self.player_index_to_choose_for == 1 and players[1]['x'] > players[0]['x']
        ):
            right = 1

        self.next_frame_to_act = self.frame + 4 + 52 + 15
        return self._buffered_action((
            (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, (left + 1) % 2, (right + 1) % 2, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, (left + 1) % 2, (right + 1) % 2, 0, 0, 1, 0),
        ))


if __name__ == "__main__":
    frames = get_ordered_numbered_images_in_folder('screenshots/jump')
    animations = get_animations_of_character('ryu')
    sprite_detector = SpriteDetector(animations)

    for index in range(len(frames)):
        print('frame ' + str(index))
        frame = frames[index]
        side = Side.LEFT
        results = sprite_detector.detect(frame, side)
        print(index, results)


    class Player:
        def __init__(self):
            self.move = None


    players = [Player(), Player()]
