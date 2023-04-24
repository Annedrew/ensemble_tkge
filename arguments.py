import argparse


def init_args():
    parser = argparse.ArgumentParser(description='Choose the methods learning the weights')

    parser.add_argument("--method", default="grid", choices=["grid", "bayes"], help='Choose the methods to learn the weights')
    parser.add_argument("--metric", default="rank", choices=["rank", "Hits@1", "Hits@3", "Hits@10", "mr", "mrr"], help="Choose the metric to calculate the ensemble score")

    return parser.parse_args()