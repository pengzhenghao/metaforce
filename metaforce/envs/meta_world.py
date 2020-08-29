import random

import gym
import metaworld
import numpy as np
from gym.spaces import Box


# compatible with metaworld a3e80c2439aa96ff221d6226bcf7ab8b49689898
# pip install git+https://github.com/rlworkgroup/metaworld.git
# @a3e80c2439aa96ff221d6226bcf7ab8b49689898

class MetaWrapperEnv(gym.Env):
    def __init__(self, env_name, eval=False):
        if env_name == "ml10":
            ml = metaworld.ML10()
        elif env_name == "ml45":
            ml = metaworld.ML45()

        self.training_envs = []
        if eval:
            task_class = ml.test_tasks
        else:
            task_class = ml.train_classes
        for name, env_cls in task_class.items():
            env = env_cls()
            task = random.choice([task for task in ml.train_tasks
                                  if task.env_name == name])
            env.set_task(task)
            self.training_envs.append(env)

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([+1, +1, +1, +1]),
        )

        self.current_env = None

    def seed(self, seed):
        # random.seed(seed)
        for env in self.training_envs:
            env.seed(seed)

    def reset(self):
        self.current_env = random.choice(self.training_envs)
        return self.current_env.reset()

    def sample(self):
        return self.current_env.action_space_sample()

    def step(self, action):
        assert self.current_env is not None, "reset before step"
        return self.current_env.step(action)

    def render(self):
        self.current_env.render()

    def set_env_state(self, state):
        self.current_env.set_env_state(state)

    def get_env_state(self):
        return self.current_env.get_env_state()

    def set_task(self, task):
        self.current_env = task

    def get_task(self):
        return self.current_env

    @property
    def observation_space(self):
        assert self.current_env is not None, "reset before step"
        obj_low = np.full(6, -np.inf)
        obj_high = np.full(6, +np.inf)
        goal_low = np.zeros(3) if self.current_env._partially_observable \
            else self.goal_space.low
        goal_high = np.zeros(3) if self.current_env._partially_observable \
            else self.goal_space.high
        return Box(
            np.hstack((self.current_env._HAND_SPACE.low, obj_low, goal_low)),
            np.hstack((self.current_env._HAND_SPACE.high, obj_high, goal_high))
        )


def make_metaworld_env_fn(env_name, ml1_name=None, eval=False):
    if env_name == "ml1":  # MT1
        def make_env():
            assert ml1_name is not None
            ml1 = metaworld.ML1(ml1_name)
            if eval:
                env = ml1.test_classes[ml1_name]()
                task = random.choice(ml1.test_tasks)
            else:
                env = ml1.train_classes[ml1_name]()
                task = random.choice(ml1.train_tasks)
            env.set_task(task)
            return env

    elif env_name == "ml10" or env_name == "ml45":  # MT10 or MT45
        def make_env():
            return MetaWrapperEnv(env_name, eval=eval)

    else:
        raise ValueError("Unknown metaworld env_name: {}".format(env_name))

    return make_env


if __name__ == '__main__':
    # For test purpose
    make_envs = [make_metaworld_env_fn("ml10", eval=False),
                 make_metaworld_env_fn("ml45", eval=False),
                 make_metaworld_env_fn("ml10", eval=True)]
    for idx, make_env in enumerate(make_envs):
        env = make_env()
        o = env.reset()
        assert env.observation_space.contains(o)
        o, r, d, i = env.step(env.action_space.sample())
        assert env.observation_space.contains(o)
        del env
        print(f"Test {idx + 1}/{4} passed. Output: \n{o}\n{r}\n{d}\n{i}")
