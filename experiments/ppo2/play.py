import os

from baselines.deepq import ppo2
from experiments.ppo2.learn import create_sf2_env

os.chdir('../..')

def main():
    env = create_sf2_env()
    observations = env.reset()
    while True:
        actions = model.
        observations, rewards, done, info = env.step(actions)
        env.render()
        if done:
            observations = env.reset()
    env.close()


if __name__ == "__main__":
    main()
