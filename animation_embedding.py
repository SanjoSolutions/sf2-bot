from Embedding import Embedding


class AnimationEmbedding(Embedding):
    pass


def create_animation_embedding(animations):
    animation_names = animations.keys()

    embedding = AnimationEmbedding(animation_names)

    return embedding
