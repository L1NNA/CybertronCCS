import json
import os
import random
import re
from concurrent.futures.process import ProcessPoolExecutor
from functools import wraps, partial
from typing import List

from Naked.toolshed.shell import muterun
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm
import numpy as np

from data_loader.DataPair import DataPair
from data_loader.Obfuscator import Obfuscator
from data_loader.tfr import read_tfr, write_tfr


def _generate_dataset(data_path: str, files: List[str], obfuscator: Obfuscator,
                      pos: int, neg: int, normalized: bool) -> List[DataPair]:
    """
    Generate all the dataset in multi-threading
    """

    def gen():
        # multi-threading
        with ProcessPoolExecutor(max_workers=50) as e:
            for extracted in e.map(partial(
                    generate_pairs,
                    data_path=data_path,
                    targets=files,
                    pos=pos,
                    neg=pos,
                    obfuscator=obfuscator,
                    normalized=normalized
            ), files):
                for p in extracted:
                    yield p

    return [p for p in tqdm(gen(), total=len(files) * pos + len(files) * neg)]


def cache_output(func):
    """
    Cache or load pairs from tf records
    """

    @wraps(func)
    def wrapper(data_path, *args, **kwargs):
        cache = str(args) + str(kwargs)
        cache = cache.strip().replace(' ', '_')
        cache = re.sub(r'(?u)[^-\w.]', '', cache)
        cache = os.path.join(data_path + '-tfr', cache)

        if os.path.exists(cache):
            ds = [
                read_tfr(os.path.join(cache, c)).map(lambda x: DataPair(**x))
                for c in sorted(os.listdir(cache), key=lambda x: int(x))
            ]
            return ds
        else:
            results = func(data_path, *args, **kwargs)
            ds = []
            for i, r in enumerate(results):
                r_folder = os.path.join(cache, str(i))
                os.makedirs(r_folder)
                ds.append(write_tfr(r, r_folder).map(lambda x: DataPair(**x)))
            return ds

    return wrapper


def _stats(pairs: List[DataPair]):
    max_seed_tokens, max_dest_tokens = 0, 0
    for pair in pairs:
        seed_tokens = int(pair['dist_token'][1])
        dest_tokens = int(pair['dist_token'][2])
        if seed_tokens > max_seed_tokens:
            max_seed_tokens = seed_tokens
        if dest_tokens > max_dest_tokens:
            max_dest_tokens = dest_tokens
    print('Max seed tokens %d, max dest tokens %d' % (max_seed_tokens, max_dest_tokens))


@cache_output
def generate(data_path: str, pos: int = 1, neg: int = 1, subset: int = None, normalized: bool = True,
             obfuscator: Obfuscator = Obfuscator.EXP_all_m1) -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
    """
    Generate the training and test dataset for Cybertron
    :param data_path: the path to the set of raw JavaScript files
    :param pos: number of replicated positive labels
    :param neg: number of replicated negative labels
    :param subset: number of the subset of the raw JavaScript files, None if take all
    :param normalized: whether to normalize the JavaScript files
    :param obfuscator: the type of obfuscation
    :return:
    """
    original_path = os.path.join(data_path, 'data')
    js_files = [
        os.path.abspath(os.path.join(original_path, f))
        for f in os.listdir(original_path) if f.endswith('.js')
    ]
    if subset is not None:
        js_files = js_files[:subset]
    
    number_of_files = len(js_files)
    print('Total JavaScript files :', number_of_files)
    np.random.shuffle(js_files)
    f_train = js_files[:int(number_of_files*0.8)]
    f_valid = js_files[int(number_of_files*0.8):int(number_of_files*0.9)]
    f_test = js_files[int(number_of_files*0.8):]

    print('Generating training set...')
    d_train = _generate_dataset(data_path, f_train, obfuscator=obfuscator,
                                pos=pos, neg=neg, normalized=normalized)
    _stats(d_train)

    print('Generating validation set...')
    d_valid = _generate_dataset(data_path, f_valid, obfuscator=obfuscator,
                                pos=pos, neg=neg, normalized=normalized)
    _stats(d_valid)

    print('Generating test set...')
    d_test = _generate_dataset(data_path, f_test, obfuscator=obfuscator,
                               pos=pos, neg=neg, normalized=normalized)
    _stats(d_test)

    return d_train, d_valid, d_test


def generate_one(seed: str, target: str, data_path: str, normalized: bool = True) -> DataPair:
    """
    Mix the seed into target by obfuscation and return a data pair containing both raw and obfuscated data
    :param seed: the seed file
    :param target: the target file
    :param data_path: the path to the index.js
    :param normalized: whether to normalize the file
    :return:
    """
    arguments = ' '.join([data_path, 'token2', seed, target, '1' if normalized else '0'])
    response = muterun('node --max-old-space-size=8192 ' + arguments)
    if response.exitcode == 0:
        result = response.stdout.decode("utf-8")
        return DataPair(**json.loads(result))
    else:
        print(response.stderr.decode("utf-8"))
        raise Exception('Unable to parse ' + seed + ' ' + target)


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


def generate_pairs(seed_file: str, data_path: str, targets: List[str], pos: int,
                   neg: int, obfuscator: Obfuscator, normalized: bool) -> List[dict]:
    """
    Randomly mix the seed file to one of the targets file
    :param seed_file: the seed file
    :param data_path: the path to the index.js
    :param targets: a list of target files
    :param pos: number of replications for positive data
    :param neg: number of replications for neg data
    :param obfuscator: the type of obfuscation
    :param normalized: whether to normalize
    :return: a list of data pairs
    """
    pairs = []
    for i in range(pos + neg):
        # retry
        j = 0
        positive = i < pos
        while j < 10:
            p1 = seed_file if positive else _choice(targets, avoid=[seed_file])
            p2 = _choice(targets, avoid=[seed_file, p1])
            arguments = [obfuscator.value, seed_file, p1, p2, '1' if positive else '0', '1' if normalized else '0']
            arguments = ' '.join(arguments)
            response = muterun('node --max-old-space-size=9216 ' + data_path + ' ' + arguments)
            if response.exitcode == 0:
                result = response.stdout.decode("utf-8")
                pairs.append(json.loads(result))
                break
            else:
                j += 1
            # throw an exception for this file
            if j == 10:
                raise Exception(seed_file + os.linesep + p1 + os.linesep + p2 +
                                os.linesep + response.stderr.decode("utf-8"))
    return pairs
