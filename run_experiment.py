import argparse
import os
from collections import deque, defaultdict

import gym
import numpy as np
import torch
# from ray.tune import track

from TD3 import TD3
from TD3_context import TD3Context
from utils import ReplayBuffer, eval_policy


# from safe_rl.td3 import TD3, TD3CTNB, ReplayBuffer, eval_policy, TD3QDiff, \
#     TD3Context
# from safe_rl.utils import core
# from safety_gym.make_env import make_env_fn

def make_env_fn(env_name):
    return lambda: gym.make(env_name)


def run_td3(
        env_fn,
        env_name,
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
        # saferl_config=None,
        # tune_track=False,
        # use_rnn=False,
        rnn_state_dim=24,
        rnn_seq_len=20,
        context_mode="disable",
        **kwargs
):
    # saferl_config = core.check_saferl_config(saferl_config)
    # max_episode_steps = saferl_config["max_ep_len"]

    # file_name = f"{env_name}_{seed}_ipd" \
    #             f"{saferl_config[core.USE_IPD]}_ipds" \
    #             f"{saferl_config[core.USE_IPD_SOFT]}_ctnb" \
    #             f"{saferl_config[core.USE_CTNB]}"

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

    # context_mode = saferl_config.get("context_mode", None)
    if context_mode == "disable":
        context_mode = None

    if context_mode:
        kwargs["context_mode"] = context_mode
        policy = TD3Context(**kwargs)
        print("We are using TD3-context model now!")
    else:
        policy = TD3(**kwargs)

    if load_model != "":
        policy_file = load_model
        policy.load(os.path.join(model_path, policy_file))

    replay_buffer = ReplayBuffer(
        state_dim, action_dim, use_rnn=bool(context_mode),
        rnn_state_dim=rnn_state_dim, rnn_seq_len=rnn_seq_len,
        rnn_use_transition="transition" in context_mode
        if context_mode else None
    )

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

    # if saferl_config[core.USE_IPD_SOFT]:
    #     ipd_episode_cost = 0.0

    last_state = np.zeros_like(state)
    last_action = np.zeros((action_dim,))

    for t in range(1, int(max_timesteps) + 1):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t <= learn_start:
            action = env.action_space.sample()
        else:

            if context_mode:
                if "transition" in context_mode:
                    transition = np.concatenate(
                        [last_state, last_action, state])
                else:
                    transition = state
                action = policy.select_action(
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

        # FIXME how to deal with early stop counting?

        # if saferl_config[core.USE_IPD] or saferl_config[core.USE_IPD_SOFT]:
        #     if saferl_config[core.USE_IPD_SOFT]:
        #         ipd_episode_cost += (
        #                 cost - saferl_config["cost_threshold"] /
        #                 saferl_config["max_ep_len"]
        #         )
        #         early_stop = ipd_episode_cost > 0.0
        #     else:
        #         early_stop = episode_cost > saferl_config["cost_threshold"]
        #     if early_stop:
        #         info["early_stop"] = True
        #         done = True
        #         early_stop_counter += 1

        # TODO why 0 when exceed the limit?
        done_bool = float(done) if episode_timesteps < max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(
            state, action, next_state, reward, done_bool, -cost,
            # rnn_state=policy.prev_state if use_rnn else None
        )
        # TODO why use nagative cost above?????

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
            eval_result = eval_policy(policy, env_fn, seed)
            evaluations.append(eval_result)
            np.save(
                os.path.join(result_path, "{}_eval".format(t)),
                evaluations
            )
            policy.save(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--policy", default="TD3", help="Policy name (TD3)"
    # )
    parser.add_argument("--env-name", default="HalfCheetah-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--learn_start",
        default=25e3,
        type=int,
        help="Time steps initial random policy is used"
    )
    parser.add_argument(
        "--eval_freq",
        default=5e3,
        type=int,
        help="How often (time steps) we evaluate"
    )
    parser.add_argument(
        "--max_timesteps",
        default=1e6,
        type=int,
        help="Max time steps to run environment"
    )
    parser.add_argument(
        "--expl_noise", default=0.1, help="Std of Gaussian exploration noise"
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Batch size for both actor and critic"
    )
    parser.add_argument("--discount", default=0.99)
    parser.add_argument(
        "--tau", default=0.005, help="Target network update rate"
    )
    parser.add_argument(
        "--policy_noise",
        default=0.2,
        help="Noise added to target policy during critic update"
    )
    parser.add_argument(
        "--noise_clip", default=0.5, help="Range to clip target policy noise"
    )
    parser.add_argument(
        "--policy_freq",
        default=2,
        type=int,
        help="Frequency of delayed policy updates"
    )
    # parser.add_argument(
    #     "--save_model", action="store_true", help="Save model and optimizer"
    # )
    parser.add_argument("--load_model", default="")
    parser.add_argument("--cost_threshold", default=25, type=int)
    parser.add_argument("--max_episode_steps", default=1000, type=int)
    parser.add_argument("--use-ipd", action="store_true")
    parser.add_argument("--use-ipd-soft", action="store_true")
    parser.add_argument("--use-ctnb", action="store_true")
    parser.add_argument("--use-qdiff", action="store_true")
    parser.add_argument("--context-mode", type=str,
                        default="add_both_transition")
    args = parser.parse_args()

    run_td3(
        env_fn=make_env_fn(args.env_name),
        # context_mode=args.context_mode,
        # saferl_config=saferl_config,
        # use_rnn=args.context_mode,
        **vars(args)
    )
