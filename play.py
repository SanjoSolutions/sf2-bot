import retro
import cv2 as cv
import numpy as np
import time

from main import get_animations_of_character, SpriteDetector, Side, ActionChooser1, \
    ActionChooser1BestResponse, KenHadokenProjectileSpriteDetector, get_animation_sprites

BUTTON_COUNT = 12


def main():
    animations = (
        get_animations_of_character('ryu'),
        get_animations_of_character('ken')
    )
    sprite_detectors = (
        SpriteDetector(animations[0]),
        SpriteDetector(animations[1]),
    )
    projectile_sprite_detectors = (
        None,
        KenHadokenProjectileSpriteDetector(
            get_animation_sprites('sprites/projectiles/ken/hadoken_projectile')
        )
    )
    action_choosers = (
        ActionChooser1(0),
        ActionChooser1BestResponse(1)
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
            results[0] = sprite_detectors[0].detect(
                observations,
                Side.LEFT if info['x_p1'] <= info['x_p2'] else Side.RIGHT
            )
            # print('Ken:')
            results[1] = sprite_detectors[1].detect(
                observations,
                Side.LEFT if info['x_p2'] <= info['x_p1'] else Side.RIGHT
            )
            # print(results)

            projectile_results = [None, None]
            projectile_results[0] = None
            projectile_results[1] = projectile_sprite_detectors[1].detect(observations)
            print(projectile_results)
        else:
            results = None
            projectile_results = None
        # action_space will by MultiBinary(16) now instead of MultiBinary(8)
        # the bottom half of the actions will be for player 1 and the top half for player 2
        actions = np.concatenate((
            action_choosers[0].choose_action(info, results, projectile_results),
            action_choosers[1].choose_action(info, results, projectile_results),
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
