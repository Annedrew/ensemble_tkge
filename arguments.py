import argparse


def init_args():
    parser = argparse.ArgumentParser(description='Choose the methods learning the weights')

    parser.add_argument('--method', default="grid" , choices=["grid", "bayes"], help='Choose the methods learning the weights')
    
    return parser.parse_args()