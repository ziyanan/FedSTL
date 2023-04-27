"""
Options for executing main file.
"""
import argparse
from ast import parse


def args_parser():
    parser = argparse.ArgumentParser()

    # main args
    parser.add_argument('--bench', type=str, default="Ditto", choices=["Ditto", "FedAvg", "FedRep", "IFCA", "FedProx"])
    parser.add_argument('--model', type=str, default="LSTM", choices=["LSTM", "GRU", "RNN", "Transformer"])
    parser.add_argument('--epoch', type=int, default=30, help="# of training epoch")
    parser.add_argument('--mode', type=str, default="train-logic", choices=['train', 'train-logic', 'eval', 'eval-sumo'])
    parser.add_argument('--dataset', type=str, default="fhwa", choices=['sumo', 'fhwa'])
    parser.add_argument('--client', type=int, default=100)
    parser.add_argument('--cluster', type=int, default=0)
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--property_type', type=str, default='constraint', help="type of property to mine", choices=['constraint', 'until', 'corr', 'infer', 'eventually'])

    # fine-tune args
    parser.add_argument('--fine_tune_iter', type=int, default=5)
    parser.add_argument('--cluster_fine_tune_iter', type=int, default=10)
    parser.add_argument('--local_updates', type=int, default=15, help="maximum number of local updates")
    parser.add_argument('--client_iter', type=int, default=10, help="# of training iterations for clients")
    
    # training args
    parser.add_argument('--batch_size', type=float, default=64, help="batch size")
    parser.add_argument('--max_lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--grad_clip', type=float, default=0.1, help="grad clip")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay")

    args = parser.parse_args()
    return args