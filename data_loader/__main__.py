import argparse
import os

import pandas as pd
import numpy as np

from data_loader.Obfuscator import obfuscators
from data_loader.generator import generate


def token_length(training, testings, name):
    file_name = f"../results/{name}_length.csv"
    
    new_results = {
        'seed_length': [],
        'dest_length': [],
        'token_ed': [],
        'token_delta': [],
        'seed_ast': [],
        'dest_ast': [],
        'ast_ed': [],
        'ast_delta': [],
        'label': []
    }

    def add(d):
        tokens = d.dist_token.numpy()
        new_results['token_ed'].append(tokens[0])
        new_results['seed_length'].append(tokens[1])
        new_results['dest_length'].append(tokens[2])
        new_results['token_delta'].append(tokens[0] / tokens[1])
        
        asts = d.dist_ast.numpy()
        new_results['ast_ed'].append(asts[0])
        new_results['seed_ast'].append(asts[1])
        new_results['dest_ast'].append(asts[2])
        new_results['ast_delta'].append(asts[0] / asts[1])
        
        new_results['label'].append(d.result.numpy()[0])

    for entry in training:
        add(entry)
    print(len(new_results['seed_length']))
    for entry in testings[0]:
        add(entry)

    df = pd.DataFrame(new_results, columns = list(new_results.keys()))
    df.to_csv(file_name)
    print(df.describe())


def entry():

    parser = argparse.ArgumentParser(
        usage='python -m data_loader',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data', choices=obfuscators.keys(),
        nargs='?', required=True,
        help='The dataset to be generated.')
    parser.add_argument(
        '--normalized',
        action='store_true',
        help='Whether to normalize the variable names.')
    parser.add_argument(
        '--duplicates',
        nargs='?', default=1, type=int, choices=range(1, 4),
        help='Number of duplicates for the dataset.')
    
    flags = parser.parse_args()
    obs = obfuscators[flags.data]
    duplicates = flags.duplicates
    training, valid, _ = generate('./data_gen', pos=duplicates, neg=duplicates, normalized=flags.normalized, obfuscator=obs)
    token_length(training, valid, flags.data)


if __name__ == '__main__':
    entry()

        
