from metaforce.utils import get_common_parser
from metaforce.run_batch_experiment import run_batch_experiment
from ray import tune

"""
Example:
    python example_pearl.py --experiment pearl --num-gpus 0 --env ant-dir --test-mode --num-seeds 1
"""

if __name__ == '__main__':
    parser = get_common_parser()
    args = parser.parse_args()
    config = dict(
        exp_name=args.exp_name,
        # env="sparse-point-robot",
        env=tune.grid_search(["ant-dir", "ant-goal", "cheetah-dir", "cheetah-vel", "humanoid-dir", "point-robot",
                              "sparse-point-robot", "walker-rand-params", "hopper-rand-params"]),
        experiment=args.experiment,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        env_name=args.env,  # cheetah-dir, ant-dir
    )
    run_batch_experiment(
        config,
        num_seeds=args.num_seeds,
        test_mode=args.test_mode,
        local_dir=args.local_dir if args.local_dir else None,
        max_num_cpus=args.max_num_cpus if args.max_num_cpus else None,
        # local_mode=True,  # use local_mode to debug, in that case all parallel is disabled.
    )
