from main import Side, get_ordered_numbered_images_in_folder, \
    KenHadokenProjectileSpriteDetector, get_animation_sprites
import cv2 as cv

animation_sprites = get_animation_sprites('sprites/projectiles/ken/hadoken_projectile')
print('animation_sprites', len(animation_sprites))
sprite_detector = KenHadokenProjectileSpriteDetector(animation_sprites)

images = get_ordered_numbered_images_in_folder('screenshots/ken_hadoken_projectile')

for index in range(len(images)):
    image = images[index]
    results = sprite_detector.detect(image)

    for result in results:
        x, y = result['location']
        sprite = result['sprite']
        sprite_image = sprite[3]
        height, width = sprite_image.shape[:2]
        cv.rectangle(image, (x, y), (x + width, y + height), (255, 255, 255), 1)

    cv.imshow('output', image)
    cv.waitKey(0)

