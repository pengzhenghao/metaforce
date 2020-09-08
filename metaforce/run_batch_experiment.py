import logging
import os
import pickle
import time

import ray
from ray import tune

from metaforce.run_experiment import run_td3_wrapper, run_pearl
from metaforce.utils import get_common_parser


def trial_name_creator(trial):
    config = trial.config
    if "env" in config:
        env_name = config["env"]
    else:
        raise ValueError("'env' term is not found in config: {}".format(config))

    if config["experiment"].lower() == "pearl":
        algo_name = "PEARL"
    elif config["experiment"].lower() == "td3_context":
        algo_name = "TD3"
        context_mode = config.get("meta_config", dict()).get("context_mode", None)
        if context_mode:
            algo_name += "-{}".format(context_mode)
    else:
        raise NotImplementedError("Unknown experiment type: {}".format(config["experiment"]))
    return "{}_{}_{}".format(algo_name, env_name, trial.trial_id)


# Formal API for user to call
def run_batch_experiment(
        config,
        num_seeds=3,
        test_mode=False,
        local_dir=None,
        max_num_cpus=None,
        seed_start=0,
        local_mode=False,
        num_samples=1
):
    # Setup commonly used config
    config.update(
        {
            "seed": tune.grid_search(
                [seed_start + i * 10 for i in range(num_seeds)]
            ),
            "test_mode": test_mode,
            "log_level": "DEBUG" if test_mode else "INFO",
            "local_dir": local_dir
        }
    )

    # Launch ray
    ray.init(
        logging_level=logging.ERROR if not test_mode else logging.DEBUG,
        log_to_driver=test_mode,
        num_cpus=max_num_cpus,
        local_mode=local_mode
    )
    if not local_dir:
        local_dir = os.path.expanduser("~/ray_results")
        config["local_dir"] = os.path.expanduser("~/ray_results")

    assert config["exp_name"], "You should specify the experiment name!"

    exp_function = {'td3_context': run_td3_wrapper, 'pearl': run_pearl}

    analysis = tune.run(
        exp_function[config['experiment'].lower()],
        name=config["exp_name"],
        config=config,
        max_failures=10 if not test_mode else 1,
        resources_per_trial={
            "cpu": config["num_cpus"],
            "gpu": config["num_gpus"],
        },
        trial_name_creator=trial_name_creator,
        verbose=2 if test_mode else 1,
        local_dir=local_dir,
        num_samples=num_samples
    )

    # save training progress as backup
    pkl_path = "{}_{}_result.pkl".format(
        config["exp_name"], time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    )
    with open(pkl_path, "wb") as f:
        data = analysis.fetch_trial_dataframes()
        pickle.dump(data, f)
        print("Result is saved at: <{}>".format(pkl_path))


if __name__ == '__main__':
    # Note: this script, which is runnable via
    # `python -m metaforce.run_batch_experiment` is used for debugging.
    # So you set whatever you like here in the config.
    # We never run this script directly when launching formal experiments.
    parser = get_common_parser()
    args = parser.parse_args()
    config = dict(
        exp_name=args.exp_name,
        env=args.env,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        experiment=args.experiment,

        batch_size=4,
        max_timesteps=10000,
        eval_freq=2000,
        learn_start=1000,
        meta_config=dict(
            context_mode=tune.grid_search(["random", "disable"])
        )
    )
    run_batch_experiment(
        config,
        test_mode=args.test_mode,
        num_seeds=args.num_seeds,
        local_dir=args.local_dir if args.local_dir else None,
        max_num_cpus=args.max_num_cpus if args.max_num_cpus else None
    )
