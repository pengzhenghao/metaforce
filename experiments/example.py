from ray import tune

from metaforce.run_batch_experiment import run_batch_experiment
from metaforce.utils import get_common_parser

if __name__ == '__main__':
    parser = get_common_parser()
    args = parser.parse_args()
    config = dict(
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        exp_name=args.exp_name,

        # Hand coded setting
        env="ml10",  # <<== handcoded
        meta_config=dict(
            context_mode=tune.grid_search([
                "add_both", "add_actor", "add_critic", "add_critic_transition",
                "add_actor_transition", "add_both_transition", "random",
                "disable"
            ]),
            rnn_state_dim=tune.grid_search([24, 48, 64])
        )
    )
    run_batch_experiment(
        config,
        test_mode=args.test_mode,
        num_seeds=args.num_seeds,
        local_dir=args.local_dir if args.local_dir else None,
        max_num_cpus=args.max_num_cpus if args.max_num_cpus else None
    )
