from main import get_animations_of_character, SpriteDetector, Side, get_ordered_numbered_images_in_folder

animations = get_animations_of_character('ken')
sprite_detector = SpriteDetector(animations)

images = get_ordered_numbered_images_in_folder('screenshots/ken_shoryuken')

for index in range(len(images)):
    image = images[index]
    side = Side.RIGHT
    results = sprite_detector.detect(image, side)
    print(index, results)
