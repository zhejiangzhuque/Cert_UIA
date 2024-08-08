from yaml import safe_load
import argparse


def get_args(config="default.yaml"):
    with open("configs/"+config, 'r') as f:
        config = safe_load(f)
    args = argparse.Namespace(**config)
    return args
