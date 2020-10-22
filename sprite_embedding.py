import itertools
from Embedding import Embedding


class SpriteEmbedding(Embedding):
    pass


def create_sprite_embedding(animations):
    sprite_names = generate_sprite_names(animations.values())

    embedding = SpriteEmbedding(sprite_names)

    return embedding


def generate_sprite_names(sprites):
    return tuple(
        generate_sprite_name_from_sprite(sprite)
        for sprite
        in itertools.chain(*sprites)
    )


def generate_sprite_name(animation_name, sprite_number):
    return animation_name + '_' + str(sprite_number)


def generate_sprite_name_from_sprite(sprite):
    return generate_sprite_name(sprite[0], sprite[1])
