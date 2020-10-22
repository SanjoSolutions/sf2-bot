from gym import Wrapper

from experiments.ppo2.SuperStreetFigher2ObservationSpaceWrapper import normalize_hp


class RoundResultOutputWrapper(Wrapper):
    def step(self, action):
        observations, rewards, done, info = super().step(action)
        if done:
            print(hp_in_percent(info['hp_p1']) + ':' + hp_in_percent(info['hp_p2']))
        return observations, rewards, done, info


def hp_in_percent(hp):
    return format_as_percent(normalize_hp(hp))


def format_as_percent(value):
    return str(int(value * 100)) + '%'
