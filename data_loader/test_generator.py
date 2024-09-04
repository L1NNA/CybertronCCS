import re
import unittest

from data_loader.generator import generate
from data_loader.tfr import make_batch


class GeneratorTests(unittest.TestCase):

    def test_file_rename(self):
        self.assertEqual(re.sub(r'(?u)[-\W.]', '_', 'abc'), 'abc')
        self.assertEqual(re.sub(r'(?u)[-\W.]', '_', '---'), '___')
        self.assertEqual(re.sub(r'(?u)[-\W.]', '_', '...'), '___')
        self.assertEqual(re.sub(r'(?u)[^-\w.]', '', '\\$$'), '')
        self.assertEqual(re.sub(r'(?u)[-\W.]', '_', 'pos_1_Obfuscator.HIGH_obf_hi'), 'pos_1_Obfuscator_HIGH_obf_hi')

    def test_generate_all(self):
        training, *testings = generate('../../data', pos=1, neg=1, normalized=False, subset=10)
        for i in make_batch(training, 2):
            print(i)
        print(training)
        print(testings)
