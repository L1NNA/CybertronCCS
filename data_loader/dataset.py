import json
import os
import random
import re
from typing import List
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from sklearn.model_selection import train_test_split
from Naked.toolshed.shell import muterun
from functools import wraps
from data_loader.Obfuscator import Obfuscator
from data_loader.DataPair import DataPair
from data_loader.tfr import write_objs_as_tfr, read_tfr_to_ds


def cache_output(func):

    @wraps(func)
    def wrapper(data_path, *args, **kwargs):
        cache = str(args) + str(kwargs)
        cache = cache.strip().replace(' ', '_')
        cache = re.sub(r'(?u)[^-\w.]', '', cache)
        cache = os.path.join(data_path+'-tfr', cache)

        if os.path.exists(cache):
            ds = [
                read_tfr_to_ds(os.path.join(cache, c)).map(lambda x: DataPair(**x))
                for c in sorted(os.listdir(cache), key=lambda x: int(x))
            ]
            return ds
        else:
            results = func(data_path, *args, **kwargs)
            ds = []
            for i, r in enumerate(results):
                r_folder = os.path.join(cache, str(i))
                os.makedirs(r_folder)
                ds.append(write_objs_as_tfr(r, r_folder).map(lambda x: DataPair(**x)))
            return ds

    return wrapper


def choice(targets: List[str], avoid: List[str]):
    assert avoid is not None and targets is not None
    assert len(targets) > len(avoid)
    while True:
        picked = random.choice(targets)
        if picked not in avoid:
            return picked


def generate_pairs(seed_file, data_path, targets: List[str], positive=True,
                   size=1, obfuscator: Obfuscator = Obfuscator.LOW, normalized=True) -> List[DataPair]:
    # random.seed(0)
    pairs = []
    for i in range(size):

        # add retry
        j = 0
        while j < 10:
            p1 = seed_file if positive else choice(targets, avoid=[seed_file])
            p2 = choice(targets, avoid=[seed_file, p1])
            arguments = ' '.join(
                [obfuscator.value, seed_file, p1, p2, '1' if positive else '0', '1' if normalized else '0'])
            response = muterun('node --max-old-space-size=9216 ' + data_path + ' ' + arguments)
            if response.exitcode == 0:
                result = response.stdout.decode("utf-8")
                # pairs.append(Pair(**json.loads(result)))
                pairs.append(json.loads(result))
                break
            else:
                j = j + 1
                if j == 10:
                    raise Exception(seed_file + os.linesep + p1 + os.linesep + p2 +
                                    os.linesep + response.stderr.decode("utf-8"))
    return pairs


def generate_dataset(data_path: str, files: List[str], obfuscator: Obfuscator = Obfuscator.LOW,
                     n_positives=1, n_negatives=1, normalized=True) -> List[DataPair]:
    def gen():
        # generate positives
        with ProcessPoolExecutor(max_workers=50) as e:
            for extracted in e.map(partial(
                generate_pairs,
                data_path=data_path,
                targets=files,
                positive=True,
                size=n_positives,
                obfuscator=obfuscator,
                normalized=normalized
            ), files):
                for p in extracted:
                    yield p
        # generate negatives:
        with ProcessPoolExecutor(max_workers=50) as e:
            for extracted in e.map(partial(
                generate_pairs,
                data_path=data_path,
                targets=files,
                positive=False,
                size=n_negatives,
                obfuscator=obfuscator,
                normalized=normalized
            ), files):
                for p in extracted:
                    yield p

    return [p for p in tqdm(gen(), total=len(files) * n_positives + len(files) * n_negatives)]


def generate_single(src_file: str, dest_file: str, data_path: str, normalized: bool = True) -> DataPair:
    arguments = ' '.join([data_path, 'token2', src_file, dest_file, '1' if normalized else '0'])
    response = muterun('node --max-old-space-size=8192 ' + arguments)
    if response.exitcode == 0:
        result = response.stdout.decode("utf-8")
        return DataPair(**json.loads(result))
    else:
        print(response.stderr.decode("utf-8"))
        raise Exception('Unable to parse ' + src_file + ' ' + dest_file)


@cache_output
def generate_all(data_path: str, pos=1, neg=1, subset: int = None, normalized: bool = True,
                 obfuscator: Obfuscator = Obfuscator.LOW):
    # we can try different thing here, and maybe pickle the whole dataset completely for reuse
    original = os.path.join(data_path, 'data')
    files = [os.path.abspath(os.path.join(original, f))
             for f in os.listdir(original) if f.endswith('.js')]
    if subset is not None:
        files = files[:subset]
    number_of_files = len(files)
    print('total files', number_of_files)
    np.random.shuffle(files)
    f_train = files[:int(number_of_files*0.8)]
    f_valid = files[int(number_of_files*0.8):int(number_of_files*0.9)]
    f_test = files[int(number_of_files*0.8):]
    print('generating training set..')
    p_train = generate_dataset(data_path, f_train, obfuscator=obfuscator,
                               n_positives=pos, n_negatives=neg, normalized=normalized)
    # same files but different pairs
    print('generating testing set op..')
    p_test_op = generate_dataset(
        data_path, f_test, obfuscator=Obfuscator.LOW, normalized=normalized)
    # different files
    # print('generating testing set hi..')
    # p_test_hi = generate_dataset(
    #     data_path, f_test, obfuscator=Obfuscator.HIGH, normalized=normalized)
    # print('generating testing set md..')
    # p_test_md = generate_dataset(
    #     data_path, f_test, obfuscator=Obfuscator.MEDIUM, normalized=normalized)
    # print('generating testing set low..')
    # p_test_lo = generate_dataset(
    #     data_path, f_test, obfuscator=Obfuscator.LOW, normalized=normalized)
    # p_test_mi = generate_dataset(data_path, f_test, obfuscator=OBF_MI)
    return p_train, p_test_op

