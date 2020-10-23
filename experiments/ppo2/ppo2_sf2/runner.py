import numpy as np
from baselines.common.runners import AbstractEnvRunner

from .sprite_ids_when_able_to_input import sprite_ids_when_able_to_input


class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, gamma, lam):
        super().__init__(env=env, model=model, nsteps=1)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.info = None
        self.idle_action = (0,) * self.env.action_space.n

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        epinfo = []

        # Given observations, get action value and neglopacs
        # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
        observations = self.obs
        actions, values, self.states, neglogpacs = self.model.step(observations, S=self.states, M=self.dones)
        mb_obs.append(observations.copy())
        mb_actions.append(actions)
        mb_values.append(values)
        mb_neglogpacs.append(neglogpacs)
        mb_dones.append(self.dones)

        total_rewards = 0

        # Take actions in env and look the results
        # Info contains a ton of useful informations
        self.obs[:], rewards, self.dones, self.info = self.env.step(actions)
        total_rewards += rewards

        while not self.dones[0] and self.info[0]['sprite_id_next_frame_p1'] not in sprite_ids_when_able_to_input:
            actions = self.idle_action
            self.obs[:], rewards, self.dones, self.info = self.env.step(actions)
            total_rewards += rewards

        for info in self.info:
            maybeepinfo = info.get('episode')
            if maybeepinfo: epinfo.append(maybeepinfo)
        mb_rewards.append(total_rewards)

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        nextnonterminal = 1.0 - self.dones
        nextvalues = last_values
        delta = mb_rewards[0] + self.gamma * nextvalues * nextnonterminal - mb_values[0]
        mb_advs[0] = delta
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfo)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


