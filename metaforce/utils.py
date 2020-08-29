import argparse


def get_common_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-mode', action="store_true")
    parser.add_argument('--algo', type=str, default='PPO')
    parser.add_argument('--num-seeds', type=int, default=3)
    parser.add_argument('--exp-name', type=str, default='')
    parser.add_argument('--num-cpus', type=int, default=1)
    parser.add_argument('--num-gpus', type=float, default=0.32)
    parser.add_argument('--max-num-cpus', type=int, default=0)
    parser.add_argument('--test-mode', action="store_true")
    parser.add_argument('--local-dir', type=str, default='.')
    return parser
