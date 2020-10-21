from main import get_animations_of_character, SpriteDetector, Side, get_ordered_numbered_images_in_folder


def detect_ken_shoryuken():
    animations = get_animations_of_character('ken')
    sprite_detector = SpriteDetector(animations)

    images = get_ordered_numbered_images_in_folder('screenshots/ken_shoryuken')

    frame_results = []
    for index in range(len(images)):
        image = images[index]
        side = Side.RIGHT
        results = sprite_detector.detect(image, side)
        frame_results.append(results)

    return frame_results


if __name__ == '__main__':
    frame_results = detect_ken_shoryuken()
    for index in range(len(frame_results)):
        frame_result = frame_results[index]
        print(index, frame_result)

