from experiments.model.train import create_sf2_env, create_model
import numpy as np

import os

os.chdir('../..')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model')


def main():
    env = create_sf2_env()
    model = create_model(env)
    data = np.load('data')
    X = data['X']
    Y = data['Y']
    model.fit(X, Y, epochs=1)
    model.save(MODEL_PATH)


if __name__ == '__main__':
    main()
