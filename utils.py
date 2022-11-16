import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr_name", type=str, default="test")
    return parser.parse_args()
