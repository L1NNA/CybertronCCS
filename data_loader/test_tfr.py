import json
import os
import unittest
import tempfile
from shutil import rmtree

from data_loader.tfr import write_objs_as_tfr, read_tfr_to_ds, make_batch, write_tfr, read_tfr


class TFRTests(unittest.TestCase):

    def setUp(self):
        self.objs = [
            {
                'att1': 0,
                'att2': [1, 2, 3, 4],
                'att3': [[1], [2, 2], [3, 3, 3]],
                'att4': [1.0, 2.0, 3.0, 4.0, 5.0],
                'att5': [
                    [[1, 1, 1]],
                    [[2, 2, 2], [2, 2, 2]],
                    [[3, 3, 3], [3, 3, 3], [3, 3, 3]]
                ],
                'att6': [[[1]], [[2]], [[3]]],
                'att7': 'abcdefg',
                'att8': ['11111', '22222']
            },
            {
                # will be detected as fix length vector of [1]
                'att1': 1,
                # will be detected as var length sequence [None]
                'att2': [2, 3, 4, 5, 6],
                # sequence of sequence [None, None]
                'att3': [[1], [2, 2], [3, 3, 3], [4, 4, 4, 4]],
                # will be detected has fixed length vector (same shape [5] for all objects)
                'att4': [1.0, 2.0, 3.0, 4.0, 5.0],
                # sequence of sequence of sequence
                'att5': [[[1, 1, 1]], [[2, 2, 2], [2, 2, 2]]],
                'att6': [[[1]], [[2]], [[3]]],
                # support string, sequence of string, sequence of sequence of string
                'att7': 'efghijk',
                'att8': ['11111', '22222', '33333']
            }
        ]

    def test_write_and_read_tfr(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            files = []
            for i, o in enumerate(self.objs):
                file = os.path.join(tmp_dir, '{}.json'.format(i))
                files.append(file)
                with open(file, 'w') as wf:
                    json.dump(o, wf)

            write_objs_as_tfr(files, 'test.tfrs')
            ds = read_tfr_to_ds('test.tfrs')
            for ind, sam in enumerate(ds):
                print("### {}".format(ind + 1))
                print(sam)

            ds = make_batch(ds, 2, pad=True)
            for ind, bat in enumerate(ds):
                print("### bat {}".format(ind + 1))
                for k, v in bat.items():
                    print('$####', k, v)
            rmtree('test.tfrs')

    def test_write_and_read_tfr_2(self):

        write_tfr(self.objs, 'test.tfrs')
        ds = read_tfr('test.tfrs')
        for ind, sam in enumerate(ds):
            print("### {}".format(ind + 1))
            print(sam)

        ds = make_batch(ds, 2, pad=True)
        for ind, bat in enumerate(ds):
            print("### bat {}".format(ind + 1))
            for k, v in bat.items():
                print('$####', k, v)
        rmtree('test.tfrs')
