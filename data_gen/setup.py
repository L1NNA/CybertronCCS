import os
from os.path import join, basename, abspath
import subprocess
import random
import multiprocessing
from typing import List
import argparse
import json
import pickle

from tqdm import tqdm

EXP = ['Exp_all', 'Exp_var', 'Exp_val', 'Exp_ast']
MIX = ['M1', 'M2', 'M3']
EXP_MIX = [(x, y) for x in EXP for y in MIX]
STAGES = ['train', 'validation', 'test']
LABELS = [0, 1]


def _choice(targets: List[str], avoid: List[str]):
    """
    Randomly pick a file from targets excluding avoid list
    """
    assert avoid is not None and targets is not None, 'Targets or avoid list cannot be None'
    assert len(targets) > len(avoid)
    while True:
        picked = random.choice(targets)
        if picked not in avoid:
            return picked

def _call_nodejs(js_file:str, arguments: List[str]):
    # Define the command to run the Node.js program
    command = ["node", "-max-old-space-size=9216", js_file, *arguments]

    # Call the Node.js program using subprocess
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check if the command was successful
    if result.returncode == 0:
        return None
    else:
        return result.stderr
    
def _join_name(target_dir, seed_path, base_path):
    seed_name = basename(seed_path)[:-3]
    base_name = basename(base_path)[:-3]
    return join(target_dir, f'{seed_name}.{base_name}.1.js')

def obfuscate_one(exp, mix, seed_path, base_files, target_dir, label=1, retries=10):
 
    error = None
    new_seed_path = seed_path

    for _ in range(retries):
        base_path = _choice(base_files, [seed_path])
        target_path = _join_name(target_dir, new_seed_path, base_path)
        arguments = [exp, mix, new_seed_path, base_path, target_path]
        error = _call_nodejs('./obfuscate.js', arguments)
        if error is None:
            # with open(seed_path, "r") as sf, open(target_path, "r") as tf:
            #     return (sf.read(), tf.read(), label)
            return

    raise Exception(error)

def obfuscate_all(input_dir, output_dir):
    """
    Generate all the dataset in multi-threading
    """
    # read all files
    all_files = [
        abspath(join(input_dir, f))
        for f in os.listdir(input_dir) if f.endswith('.js')
    ]

    # train val test split
    random.shuffle(all_files)
    total_length = len(all_files)
    ep_length = int(total_length * 0.8)
    np_length = int(total_length * 0.9)

    train_files = all_files[:ep_length]
    val_files = all_files[ep_length:np_length]
    test_files = all_files[np_length:]

    # for train, val and testing
    for stage, files in zip(STAGES, [train_files, val_files, test_files]):
        # for each experiment
        for exp, mix in EXP_MIX:
            cur_output_dir = join(output_dir, stage, f'{exp}_{mix}')
            os.makedirs(cur_output_dir, exist_ok=True)
            with multiprocessing.Pool() as pool:
                pairs = [(exp, mix, f, files, cur_output_dir, 1) for f in files]
            
                desc = f'Obfuscation {exp}_{mix}_{stage}'
                for _ in tqdm(pool.starmap(obfuscate_one, pairs), total=len(pairs), desc=desc):
                    pass
                # with open(join(cur_output_dir, desc+'.pickle'), 'wb') as f:
                #     pickle.dump(results, f)

            with open(join(cur_output_dir, desc+'_sources.txt'), 'w') as f:
                file_names = [basename(f) for f in files]
                f.write('\n'.join(file_names))

def _traverse_experiments(input_dir, output_dir, is_source_dir=False):
    if is_source_dir:
        yield [
            (abspath(join(input_dir, f)), abspath(join(output_dir, f))) 
            for f in os.listdir(input_dir) if f.endswith('.js')
        ], input_dir
        return
    for stage in STAGES:
        for exp, mix in EXP_MIX:
            cur_input_dir = join(input_dir, stage, f'{exp}_{mix}')
            cur_output_dir = join(output_dir, stage, f'{exp}_{mix}')
            os.makedirs(cur_output_dir, exist_ok=True)
            yield [
                (abspath(join(cur_input_dir, f)), abspath(join(cur_output_dir, f))) 
                for f in os.listdir(cur_input_dir) if f.endswith('.js')
            ], f'{exp}_{mix}_{stage}'

def deobfuscate_one(input_path, output_path):
    result = _call_nodejs('./deobfuscate.js', [input_path, output_path])
    if result is not None:
        print('Failed to deobfuscate ', input_path)
        
def deobfuscate_all(input_dir, output_dir):

    for pairs,desc in _traverse_experiments(input_dir, output_dir):
        desc = 'Deobfuscation ' + desc
        with multiprocessing.Pool() as pool:
            for _ in tqdm(pool.starmap(deobfuscate_one, pairs), total=len(pairs), desc=desc):
                pass

def ast2vec_one(input_path, output_path):
    output_path += 'on'
    result = _call_nodejs('./AST2Vec.js', [input_path, output_path])
    if result is not None:
        raise Exception(result)

def ast2vec(input_dir, output_dir, is_source_dir):
        
    for pairs,desc in _traverse_experiments(input_dir, output_dir, is_source_dir):
        desc = 'AST2Vec ' + desc
        with multiprocessing.Pool() as pool:
            for _ in tqdm(pool.starmap(ast2vec_one, pairs), total=len(pairs), desc=desc):
                pass

def pair_all(_, obs_dir):

    for stage in STAGES:
        for exp, mix in EXP_MIX:
            cur_obs_dir = join(obs_dir, stage, f'{exp}_{mix}')
            obs_files = [f for f in os.listdir(cur_obs_dir) if f.endswith('.js')]
            sources = [f.split('.')[0]+'.js' for f in obs_files]
            bases = [f.split('.')[1]+'.js' for f in obs_files]
            counters = [_choice(sources, [sources[i], bases[i]]) for i in range(len(sources))]
            pairs = []
            for i in range(len(obs_files)):
                for label, seeds in enumerate([counters, sources]):
                    # TODO find close seeds
                    pairs.append({
                        'seed': seeds[i],
                        'target': obs_files[i],
                        'label': label,
                    })
            with open(join(cur_obs_dir, 'pairs.pickle'), 'wb') as f:
                pickle.dump(pairs, f)

def clean_one(input_file, output_file):
    result = _call_nodejs('./clean.js', [input_file, output_file])
    if result is not None:
        raise Exception(result)
            
def clean_all(input_dir, output_dir, is_source_dir):
    for pairs,desc in _traverse_experiments(input_dir, output_dir, is_source_dir):
        desc = 'Clean all ' + desc
        with multiprocessing.Pool() as pool:
            for _ in tqdm(pool.starmap(clean_one, pairs), total=len(pairs), desc=desc):
                pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--action', required=True, default='obfuscation',
                        choices=['obfuscation', 'deobfuscation', 'ast2vec', 'pair_all', 'clean'],
                        help='action')
    parser.add_argument('--input_path', type=str, required=True,
                        help='input directory or file')
    parser.add_argument('--is_source', action='store_true',
                        help='if not to traverse the input directory')
    parser.add_argument('--output_path', type=str, required=True,
                        help='input directory or file')
    
    args = parser.parse_args()

    if args.action == 'obfuscation':
        obfuscate_all(args.input_path, args.output_path)
    elif args.action == 'deobfuscation':
        deobfuscate_all(args.input_path, args.output_path)
    elif args.action == 'ast2vec':
        ast2vec(args.input_path, args.output_path, args.is_source)
    elif args.action == 'clean':
        clean_all(args.input_path, args.output_path, args.is_source)
    else:
        pair_all(args.input_path, args.output_path)