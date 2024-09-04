from typing import List
import os, json
from os.path import isfile, join, exists
import pickle

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Input, Dot
from tensorflow.keras.models import Model

from data_loader.DataPair import DataPair
from data_loader.dataset import generate_single
from data_loader.tfr import make_batch


class BaseModel:

    def __init__(self, name, max_seq_len, max_vocab=2500, **kwargs):
        self._name = name
        self._cos_model = None
        self._history = None
        self._tokenizer = None
        self.max_seq_len = max_seq_len
        self.max_vocab = max_vocab
        self.built = False
        self.vectorizer = None
        self.model_path = None
        self.early_stopping = True

    def model_summary(self):
        self._cos_model.summary()

    def build(self):
        """
        Need to have a build method, since some params, such as vocab_size,
        is only available after preprocessing of the training set. It is not
        available in the init function
        """
        pass

    def _build_vectorizer(self):
        self.vectorizer = TextVectorization(
            standardize=None,
            max_tokens=self.max_vocab, output_sequence_length=self.max_seq_len
        )

    def preprocess(self, pairs: tf.data.Dataset, train=False, batch_size=64):
        """
        Default preprocessor that extracts tokens and apply a static tokenizer on the
        JavaScript file. A model can overwrite this if necessary.

        Args:
            pairs (tf.data.Dataset): [List of pairs of javascript code]
            train (bool, optional): [It is a training set or not]. Defaults to False.
            batch_size (int, optional): [number of batch size]. Defaults to 64.
        """

        def _mapper(x: DataPair):
            return {
                'obf': x.obfuscated_tokens,
                'org': x.seed_tokens,
                'lbl': x.result
            }

        pairs = pairs.map(_mapper)

        if train and self.vectorizer is None:
            def _v_mapper(x):
                return tf.concat([
                    x['obf'],
                    x['org']
                ], axis=0)

            self._build_vectorizer()
            self.vectorizer.adapt(pairs.map(_v_mapper))

        pairs = make_batch(pairs, batch_size) if batch_size > 1 else pairs
        pairs = pairs.map(lambda x: ({
                                         'obf': self.vectorizer(x['obf']),
                                         'org': self.vectorizer(x['org']),
                                     }, x['lbl']))
        pairs._preprocessed = True
        return pairs

    def train(self, train: tf.data.Dataset, valid: List[tf.data.Dataset], epochs, batch_size=64, log_dir=None):
        # typical flow: preprocess => build => fit => evaluate

        # Preprocess the data
        train_ds = self.preprocess(train, train=True, batch_size=batch_size)
        valid_ds = self.preprocess(valid, batch_size=batch_size)

        # Build model
        if not self.built:
            self.build()
            self.built = True
        self.model_summary()

        history = os.path.join(self.model_path, 'history.csv')
        
        def track_history(epoch, logs):
            with open(history, 'a') as af:
                af.write(json.dumps(logs)+'\n')
                
        callbacks = [
            # tf.keras.callbacks.LambdaCallback(
            # on_epoch_end=lambda e, l: self.evaluate(valid_ds_all[1:])),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=track_history)
        ]
        
        if self.early_stopping:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=30, restore_best_weights=True)
            )
        if log_dir is not None:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            callbacks.append(tensorboard_callback)
        
        self._history = self._cos_model.fit(
            train_ds, epochs=epochs, batch_size=batch_size,
            validation_data=valid_ds,
            callbacks=callbacks)

    def evaluate(self, dataset: tf.data.Dataset, verbose=0):
        ds_test = self.preprocess(dataset)
        print(self._cos_model.evaluate(ds_test, verbose=verbose))

    def predict_one(self, src_file, dest_file, verbose=0, data='../data', normalized=True):
        pair = generate_single(src_file, dest_file, data, normalized)
        input_pair = {
            'obf': self.vectorizer(np.array([pair.obfuscated_tokens])),
            'org': self.vectorizer(np.array([pair.seed_tokens]))
        }
        print(self._cos_model.predict(input_pair, verbose=verbose)[0][0])
        
    def scan(self, seeds, dest_file):
        raise 'Method not implemented'

    def plot(self):
        if self._history is None:
            return
        plt.plot(self._history.history['accuracy'])
        plt.plot(self._history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.ylim([0.5, 1])
        plt.show()

    def save(self, path):
        tokenizer_path = join(path, self._name + '.tkn.pkl')
        with open(tokenizer_path, 'wb') as wbf:
            pickle.dump(self.vectorizer.get_weights(), wbf)

        weights_path = join(path, self._name)
        self._cos_model.save_weights(weights_path)

    def load(self, path):
        if not exists(path):
            os.makedirs(path)
        self.model_path = path
        index_file = join(path, self._name + ".index")
        if isfile(index_file):
            tokenizer_path = join(path, self._name + '.tkn.pkl')
            with open(tokenizer_path, 'rb') as wbf:
                self._build_vectorizer()
                weights = pickle.load(wbf)
                self.vectorizer.set_weights(weights)

            if not self.built:
                self.build()
                self.built = True

            weights_path = join(path, self._name)
            self._cos_model.load_weights(weights_path)
            
    def _cos_sim(self, input_length, name='cos_sim'):
        i_source = Input(shape=(input_length,))
        i_dest = Input(shape=(input_length,))
        # Build Cosine Similarity
        cos_sim = Dot(axes=1, normalize=True)([i_source, i_dest])
        predict = (cos_sim + 1) / 2
        predict = tf.clip_by_value(predict, 0, 1)
        return Model([i_source, i_dest], predict, name=name)
