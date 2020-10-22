from gym import Wrapper


class MonitorWrapper(Wrapper):
    def step(self, action):
        result = super().step(action)
        super().render(mode='human')
        return result

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        super().render(mode='human')
        return result
