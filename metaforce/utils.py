import argparse


def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            if k not in to:
                to[k] = v
            else:
                deep_update_dict(v, to[k])
        else:
            if k not in to:
                to[k] = v
    return to


def get_common_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default="td3_context")  # pearl, td3_context
    parser.add_argument('--local-mode', action="store_true")
    parser.add_argument('--num-seeds', type=int, default=3)
    parser.add_argument('--exp-name', type=str, default='')
    parser.add_argument('--num-cpus', type=int, default=1)
    parser.add_argument('--num-gpus', type=float, default=0.32)
    parser.add_argument('--max-num-cpus', type=int, default=0)
    parser.add_argument('--test-mode', action="store_true")
    parser.add_argument('--local-dir', type=str, default='.')

    # ==========td3_context==========
    parser.add_argument('--algo', type=str, default='PPO')
    parser.add_argument("--env", type=str, default="BipedalWalker-v3")

    # ==========pearl==========
    # parser.add_argument("--env-config", type=str, default=None)

    return parser
