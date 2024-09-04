import argparse
import os
import pathlib
import json

import numpy as np

from data_loader.Obfuscator import obfuscators
from data_loader.generator import generate
from cybertron import ALL_MODELS


def parse_json(json_string):
    if json_string is None:
        return {}
    try:
        return json.loads(json_string)
    except:
        return {}


def entry():

    parser = argparse.ArgumentParser(
        usage='python -m cybertron',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', choices=ALL_MODELS.keys(),
        nargs='?', required=True,
        help='The model to run.')
    parser.add_argument(
        '--name', nargs='?',
        help='The name of the model.')
    parser.add_argument(
        '--device',
        nargs='?', default=0, type=int, choices=range(3),
        help='The gpu device to use.')
    parser.add_argument(
        '--args', nargs='?',
        help='Arguments in json')
    
    subparsers = parser.add_subparsers(help='sub-command help', dest='command')
    
    # Train the designed experiments
    parser_train = subparsers.add_parser('train', help='The model training process')
    parser_train.add_argument(
        '--data', choices=obfuscators.keys(),
        nargs='?', required=True,
        help='The dataset to be test.')
    parser_train.add_argument(
        '--epoch',
        nargs='?', default=20, type=int,
        help='Number of epochs to train.')
    
    # Train the custom seeds folder
    parser_ctrain = subparsers.add_parser('ctrain', help='The model custom training process')
    parser_ctrain.add_argument(
        '--seeds', required=True,
        nargs='?', type=pathlib.Path,
        help='The path to the seed folder.')
    parser_ctrain.add_argument(
        '--weight', required=True,
        nargs='?', type=pathlib.Path,
        help='The path to the weights folder of the model')
    parser_ctrain.add_argument(
        '--epoch',
        nargs='?', default=20, type=int,
        help='Number of epochs to train.')
    
    # Clone search the dest file from the seeds folder
    parser_test = subparsers.add_parser('test', help='The model test process')
    parser_test.add_argument(
        '--weight', required=True,
        nargs='?', type=pathlib.Path,
        help='The path to the weights folder of the model')
    parser_test.add_argument(
        '--seeds', required=True,
        nargs='?', type=pathlib.Path,
        help='The path to the seed folder')
    parser_test.add_argument(
        '--dest', required=True,
        nargs='?', type=argparse.FileType('r', encoding='UTF-8'),
        help='The path to the dest file')

    
    flags = parser.parse_args()
    
    if flags.command in ['test', 'ctrain']:
        command = parser_ctrain if flags.command == 'ctrain' else parser_test
        if not os.path.isdir(flags.seeds):
            command.print_help()
            command.error('seeds path does not exist')
            return
        if not os.path.isdir(flags.weight):
            command.print_help()
            command.error('weight path does not exist')
            return
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(flags.device)
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    seq_len = 6000
    model_name = flags.name if flags.name else flags.model
    
    model_cls = ALL_MODELS[flags.model]
    args = parse_json(flags.args)
    args['max_seq_len'] = seq_len
    model = model_cls(**args)
    
    if flags.command == 'train':
        obs = obfuscators[flags.data]
        data_model_path = f'{flags.data}/{model_name}_{seq_len}'
        file_path = '../results/' + data_model_path
        training, valid, testings = generate('./data_gen', pos=1, neg=1, normalized=False, obfuscator=obs)
        model.load(file_path)
        log_dir = '../logs/' + data_model_path
        model.train(training, valid, epochs=flags.epoch, batch_size=64, log_dir=log_dir)
        model.evaluate(testings)
        model.save(file_path)
    elif flags.command == 'test':
        model.load(flags.weight)
        model.scan(flags.seeds, flags.dest)
    elif flags.command == 'ctrain':
        model.load(flags.weight)
        # TODO generate training sets based on seeds
        model.train(training, testings, epochs=flags.epoch, batch_size=64)
        model.evaluate(testings)
        model.save(flags.weight)


if __name__ == '__main__':
    entry()