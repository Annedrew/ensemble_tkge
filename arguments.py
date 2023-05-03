import argparse


def init_args():
    parser = argparse.ArgumentParser(description='Choose the methods learning the weights')

    parser.add_argument("--method", default="grid", choices=["grid", "bayes"])
    parser.add_argument("--metric", default="rank", choices=["rank", "Hits@1", "Hits@3", "Hits@10", "mr", "mrr", "rr"])
    # parser.add_argument("--ens_num", default=5, choices=[1,2,3,4,5])

    return parser.parse_args()