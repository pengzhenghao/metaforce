import argparse
import os
import tempfile
from collections import deque, defaultdict

import gym
import numpy as np
import torch
from ray.tune import track

from metaforce.envs import make_env_fn
from metaforce.mql import TD3Context
from metaforce.mql.utils import LearnableContextReplayBuffer, \
    RandomContextReplayBuffer
from metaforce.td3 import TD3, ReplayBuffer, eval_policy


def check_meta_config(meta_config):
    assert "context_mode" in meta_config
    assert meta_config["context_mode"] in [
        "add_both", "add_actor", "add_critic", "add_critic_transition",
        "add_actor_transition", "add_both_transition", "random", "disable"
    ]
    meta_config["rnn_use_transition"] = \
        "transition" in meta_config["context_mode"]
    if meta_config["context_mode"] == "disable":
        meta_config["context_mode"] = None
    if "rnn_state_dim" not in meta_config:
        meta_config['rnn_state_dim'] = 24
    return meta_config


def run_td3(
        env_fn,
        # env_name,
        seed,
        local_dir=".",
        load_model="",
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        expl_noise=0.1,
        noise_clip=0.5,
        policy_freq=2,
        eval_freq=5e3,
        batch_size=256,
        learn_start=25e3,
        max_timesteps=1e6,
        max_episode_steps=1000,
        eval_env_fn=None,
        meta_config=None,
        tune_track=False,
        # use_rnn=False,
        rnn_state_dim=24,
        rnn_seq_len=20,
        # context_mode="disable",
        **kwargs
):
    # Define those variable for meta learning
    meta_config = check_meta_config(meta_config)
    context_mode = meta_config.get("context_mode")

    result_path = os.path.join(local_dir, "results")
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    model_path = os.path.join(local_dir, "models")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    env = env_fn()

    # Set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if isinstance(env.observation_space, gym.spaces.Dict):
        # We are observing RGB image now!
        # state_dim = env.observation_space["vision"].shape
        state_dim = (84, 84, 3)  # Hard-coded
    else:
        state_dim = env.observation_space.shape[0]

    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = dict(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=discount,
        tau=tau,
    )

    # Initialize policy
    kwargs["policy_noise"] = policy_noise * max_action
    kwargs["noise_clip"] = noise_clip * max_action
    kwargs["policy_freq"] = policy_freq

    if context_mode:
        kwargs["context_mode"] = context_mode
        kwargs["hidden_state_dim"] = meta_config.get("rnn_state_dim", 24)
        policy = TD3Context(**kwargs)
        print("We are using TD3-context model now!")
    else:
        policy = TD3(**kwargs)

    if load_model != "":
        policy_file = load_model
        policy.load(os.path.join(model_path, policy_file))

    if context_mode == "random":
        replay_buffer = RandomContextReplayBuffer(
            state_dim, action_dim,
            context_dim=meta_config["rnn_state_dim"]
        )
    elif context_mode is not None:
        replay_buffer = LearnableContextReplayBuffer(
            state_dim, action_dim,
            rnn_seq_len=meta_config.get("rnn_seq_len", 20),
            rnn_state_dim=meta_config["rnn_state_dim"],
            rnn_use_transition=meta_config["rnn_use_transition"]
        )
    else:
        replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, eval_env_fn or env_fn, seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_cost = 0
    episode_timesteps = 0
    episode_num = 0
    prev_timesteps_total = 0
    prev_episode_num = 0
    early_stop_counter = 0
    cumulative_cost = 0

    record_length = int(eval_freq / max_episode_steps)
    episode_reward_record = deque(maxlen=record_length)
    episode_cost_record = deque(maxlen=record_length)
    cost_rate_record = deque(maxlen=record_length)
    episode_length_record = deque(maxlen=record_length)
    cumulative_cost_record = deque(maxlen=record_length)
    other_stats = defaultdict(lambda: deque(maxlen=record_length))

    last_state = np.zeros_like(state)
    last_action = np.zeros((action_dim,))

    for t in range(1, int(max_timesteps) + 1):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t <= learn_start:
            action = env.action_space.sample()
            if context_mode == "random":
                context = np.zeros((policy.hidden_state_dim,), dtype=np.float32)
        else:

            if context_mode:
                if ("transition" in context_mode) or (context_mode == "random"):
                    transition = np.concatenate(
                        [last_state, last_action, state])
                else:
                    transition = state
                action, context = policy.select_action(
                    np.array(state), np.array(transition))
            else:
                action = policy.select_action(np.array(state))

            action = (
                    action +
                    np.random.normal(0, max_action * expl_noise,
                                     size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, info = env.step(action)
        cost = info.get('cost', 0.0)

        episode_cost += cost
        cumulative_cost += cost

        # TODO why 0 when exceed the limit?
        done_bool = float(done) if episode_timesteps < max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer_add = (
            state, action, next_state, reward, done_bool, -cost
        )
        if context_mode == "random":
            replay_buffer_add += (context,)
        elif context_mode is not None:
            replay_buffer_add += (policy.prev_state.detach(),)
        replay_buffer.add(*replay_buffer_add)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t > learn_start:
            stats = policy.train(replay_buffer, batch_size)
            if stats:
                assert isinstance(stats, dict)
                for k, v in stats.items():
                    assert not isinstance(v, torch.Tensor)
                    other_stats[k].append(v)

        if done:
            cost_rate = cumulative_cost / t
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will
            # increment +1 even if done=True
            print(
                f"Total T: {t} Episode Num: {episode_num + 1} Episode T: "
                f"{episode_timesteps} Reward: {episode_reward:.3f} Cost:"
                f"{episode_cost:.3f} Early Stop Num: {early_stop_counter:.3f}"
                f" Cumulative Cost: {cumulative_cost:.3f} "
                f"Cost Rate: {cost_rate:.3f}"
            )
            episode_cost_record.append(episode_cost)
            episode_reward_record.append(episode_reward)
            cost_rate_record.append(cost_rate)
            cumulative_cost_record.append(cumulative_cost)
            episode_length_record.append(episode_timesteps)

            # Reset environment
            state, done = env.reset(), False
            episode_cost = 0
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            last_state = np.zeros_like(state)
            last_action = np.zeros((action_dim,))

            # if saferl_config[core.USE_IPD_SOFT]:
            #     ipd_episode_cost = 0

        else:
            last_action = action.copy()
            last_state = state.copy()

        # Evaluate episode
        if t % eval_freq == 0:
            # if True:
            eval_result = eval_policy(policy, env_fn, seed)
            evaluations.append(eval_result)
            np.save(
                os.path.join(result_path, "{}_eval".format(t)),
                evaluations
            )
            policy.save(os.path.join(model_path, "{}".format(t)))

            if tune_track and t > learn_start:
                track.log(
                    episode_reward_min=np.min(episode_reward_record),
                    episode_reward_mean_train=np.mean(episode_reward_record),
                    episode_reward_mean=eval_result[0],
                    episode_reward_max=np.max(episode_reward_record),
                    episode_cost_min=np.min(episode_cost_record),
                    episode_cost_mean_train=np.mean(episode_cost_record),
                    episode_cost_mean=eval_result[1],
                    episode_cost_max=np.max(episode_cost_record),
                    cost_rate=np.mean(cost_rate_record),
                    early_stop_rate=early_stop_counter / episode_num,
                    episodes_this_iter=episode_num - prev_episode_num,
                    timesteps_this_iter=t - prev_timesteps_total,
                    mean_loss=np.mean(episode_cost_record),
                    mean_accuracy=np.mean(cost_rate_record),
                    **{k: np.mean(v)
                       for k, v in other_stats.items()}
                )
                prev_episode_num = episode_num
                prev_timesteps_total = t


def run_td3_wrapper(config):
    # Check meta config
    meta_config = config.get("meta_config", {})
    if not meta_config:
        print("[WARNING] The input meta config is empty!")

    # Build make_env function
    env_fn = make_env_fn(env_name=config["env"], eval=False)
    eval_env_fn = make_env_fn(env_name=config["env"], eval=True)

    # Modify a set of config to accelerate debugging.
    kwargs = dict() if not config["test_mode"] else dict(
        learn_start=1000, max_timesteps=5000, eval_freq=2000
    )
    if config.get("max_timesteps", False):
        kwargs["max_timesteps"] = config["max_timesteps"]

    # Get local directory
    local_dir = config["local_dir"]
    exp_name = config["exp_name"]
    assert local_dir
    local_dir = os.path.join(local_dir, exp_name)

    run_td3(
        env_fn=env_fn,
        seed=config["seed"],
        local_dir=local_dir,
        meta_config=config.get("meta_config", None),
        tune_track=True,
        eval_env_fn=eval_env_fn,
        **kwargs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="HalfCheetah-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--load_model", default="")
    parser.add_argument("--context-mode", type=str,
                        default="add_both_transition")
    args = parser.parse_args()

    config = dict(
        exp_name="DELETE_ME",
        local_dir=tempfile.gettempdir(),
        test_mode=True,
        batch_size=32,
        max_timesteps=10000,
        eval_freq=2000,
        learn_start=1000,
        seed=0,
        env=args.env,
        meta_config=dict(
            context_mode=args.context_mode
        )
    )

    run_td3_wrapper(config)
