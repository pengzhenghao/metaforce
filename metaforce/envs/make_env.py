import gym

from metaforce.envs.mujoco_envs import ENVS
from metaforce.envs.mujoco_envs.wrappers import NormalizedBoxEnv
from metaforce.envs.meta_world import make_metaworld_env_fn
import os
import json


class GymWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env=env)

    def set_env_state(self, state):
        self.env.sim.set_state(state)

    def get_env_state(self):
        return self.env.sim.get_state()


def make_env_fn(env_name, eval=False):
    assert env_name != "mujoco_meta", "Deprecated environment name mujoco_meta!"
    if env_name in ["ml1", "ml10", "ml45"]:
        return make_metaworld_env_fn(env_name, eval=eval)
    elif env_name in ["ant-dir", "ant-goal", "cheetah-dir", "cheetah-vel", "humanoid-dir", "point-robot",
                      "sparse-point-robot", "walker-rand-params", "hopper-rand-params"]:
        env_config_path = os.path.join(os.path.dirname(__file__), "configs/{}.json".format(env_name))
        with open(os.path.join(env_config_path)) as f:
            env_params = json.load(f)["env_params"]

        def _make():
            return NormalizedBoxEnv(ENVS[env_name](**env_params))
    else:
        def _make():
            env = gym.make(env_name)
            return GymWrapper(env)

    return _make
