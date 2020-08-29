import gym

from metaforce.envs.meta_world import make_metaworld_env_fn


class GymWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env=env)

    def set_env_state(self, state):
        self.env.sim.set_state(state)

    def get_env_state(self):
        return self.env.sim.get_state()


def make_env_fn(env_name, eval=False):
    if env_name in ["ml1", "ml10", "ml45"]:
        return make_metaworld_env_fn(env_name, eval=eval)
    else:
        def _make():
            env = gym.make(env_name)
            return GymWrapper(env)

        return _make
