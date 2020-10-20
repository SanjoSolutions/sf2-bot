import retro
import cv2 as cv
import numpy as np
import time

from main import get_animations_of_character, SpriteDetector, Side, ActionChooser


BUTTON_COUNT = 12


def main():
    animations = (
        None, # get_animations_of_character('ryu'),
        get_animations_of_character('ken')
    )
    sprite_detectors = (
        None, # SpriteDetector(animations[0]),
        SpriteDetector(animations[1]),
    )
    action_choosers = (
        ActionChooser(0),
        None
    )

    retro.data.Integrations.add_custom_path('/Applications')
    env = retro.make(
        game='SuperStreetFighter2-Snes',
        inttype=retro.data.Integrations.ALL,
        state='ryu_vs_ken_both_controlled',
        players=2
    )
    info = None
    obs = env.reset()
    FPS = 30
    duration_per_frame = 1 / FPS  # in seconds
    while True:
        frame_start_time = time.time()
        if info is not None:
            observations = cv.cvtColor(obs, cv.COLOR_RGB2GRAY)
            results = [None, None]
            # print('Ryu:')
            # results[0] = sprite_detectors[0].detect(
            #     observations,
            #     Side.LEFT if info['x_p1'] <= info['x_p2'] else Side.RIGHT
            # )
            results[0] = None
            # print('Ken:')
            side = Side.LEFT if info['x_p2'] <= info['x_p1'] else Side.RIGHT
            results[1] = sprite_detectors[1].detect(
                observations,
                side
            )
            print(results)
        else:
            results = None
        # action_space will by MultiBinary(16) now instead of MultiBinary(8)
        # the bottom half of the actions will be for player 1 and the top half for player 2
        # chosen_actions = np.concatenate((
        #     choose_action(0, info, results),
        #     choose_action(1, info, results)
        # ))
        random_actions = env.action_space.sample()
        actions = np.concatenate((
            action_choosers[0].choose_action(info, results),
            random_actions[(BUTTON_COUNT + 1):(BUTTON_COUNT + 1 + BUTTON_COUNT)]
        ))
        obs, rew, done, info = env.step(actions)
        # rew will be a list of [player_1_rew, player_2_rew]
        # done and info will remain the same
        env.render()
        # input("Press Enter to continue...")

        frame_end_time = time.time()
        frame_duration = frame_end_time - frame_start_time
        frame_sleep_duration = duration_per_frame - frame_duration
        if frame_sleep_duration > 0:
            time.sleep(frame_sleep_duration)

        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
