import sys, os
from os.path import isfile, join

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.python.keras.layers.recurrent import _generate_zero_filled_state_for_cell

from cybertron.BaseModel import BaseModel


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)


class LeapRNNLayer(Layer):
    """ Leap RNN with GRU cell instead. Adopted from:
    https://www.ijcai.org/Proceedings/2019/0697.pdf"""

    def __init__(self, hidden_dim, neighbours=64, rnn_cell_dim=None, name='LeapRNNLayer', **kwargs):
        super(LeapRNNLayer, self).__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim
        self.rnn_cell_dim = rnn_cell_dim if rnn_cell_dim else hidden_dim
        self.neighbours = neighbours
        self.rnn_cell = GRUCell(self.rnn_cell_dim)
        self.f_after = None
        self.f_skip = None
        self.built = False

    def build(self, input_shape):
        if not self.built:
            self.f_after = self.add_weight(
                name=self.name+'_f_after',
                shape=[self.neighbours, input_shape[-1], self.hidden_dim],
                initializer=tf.initializers.GlorotUniform(),
                dtype=tf.float32
            )
            self.f_skip = self.add_weight(
                name=self.name+'_f_skip',
                shape=[self.rnn_cell_dim + self.hidden_dim, 2],
                initializer=tf.initializers.GlorotUniform(),
                dtype=tf.float32
            )
            self.built = True

        cell = self.rnn_cell
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if not cell.built:
            with K.name_scope(cell.name):
                cell.build(input_shape)
                cell.built = True

    # Used when this layer is part of a network
    # o_inputs is of shape [batch, seq, embedding_dim]
    def call(self, o_inputs, **_):
        seq_size = tf.shape(o_inputs)[1]

        # context after time step t (pre-computed using CNN)
        # the list of integers denote the movement step on each direction
        # we should move 1 at a time, but the window size will be 3 (inferred from f_after)
        z_after = tf.nn.conv1d(
            o_inputs, self.f_after, [1, 1, 1], 'SAME'
        )
        z_after = tf.tanh(z_after)
        # [batch, seq, embedding_dim]

        def lp_fn(t, ht_1, res):
            # run for each timestamp
            # t: integer, current time stamp
            # ht_1: the last state
            # res: the TensorArray to hold result
            # z_after = inputs[:, t, :]
            z_before = ht_1
            z = tf.concat([z_before, z_after[:, t, :]], axis=-1) @ self.f_skip
            skip = gumbel_softmax_sample(z)
            # skip [None, 1]/[None]
            # in paper: skip [None, 2]

            # gru_cell expect the state to be a list of tensors
            ht, _ = self.rnn_cell(o_inputs[:, t, :], [ht_1])
            # ht = (skip * ht_1 + (1 - skip) * ht)
            skip = tf.expand_dims(skip, -1)
            ht = (skip[:, 0, :] * ht_1 + skip[:, 1, :] * ht)

            # write result for timestamp t
            res = res.write(t, ht)
            return t+1, ht, res

        # initial zero state
        h0 = _generate_zero_filled_state_for_cell(
            self.rnn_cell, o_inputs, None, None)
        
        outputs_a = tf.TensorArray(dtype=tf.float32, size=seq_size)

        * _, outputs_a = tf.while_loop(
            lambda t, *_: t < seq_size,
            lp_fn,
            (0, h0, outputs_a)
        )
        outputs = outputs_a.stack()
        outputs = tf.transpose(outputs, [1, 0, 2])
        return outputs


class LeapGRU(BaseModel):
    def __init__(self, max_seq_len, name=None):
        super(LeapGRU, self).__init__(
            max_seq_len=max_seq_len,
            name=name if name else 'LeapGRU')

    def build(self):
        input_dim = len(self.vectorizer.get_vocabulary())
        input_length = self.max_seq_len

        # Build source model
        source_input = Input(shape=(input_length,))
        dest_input = Input(shape=(input_length,))
        embedding_layer = Embedding(
            input_dim=input_dim, output_dim=128, input_length=input_length)

        h3 = LeapRNNLayer(128, name='h2')
        h4 = LeapRNNLayer(128, name='h3')
        dense_layer = Dense(128)
        l_cos_sim = self._cos_sim(128)

        def encode(any_input, name):
            with tf.name_scope(name):
                embedded = embedding_layer(any_input)
                embedded_rev = K.reverse(embedded, axes=1)
                fwd = h3(embedded)[:, -1, :]
                bwd = h4(embedded_rev)[:, -1, :]
                z = dense_layer(tf.concat([fwd, bwd], axis=-1))
                return z

        source_output = encode(source_input, name='src')
        dest_output = encode(dest_input, name='des')

        # Build Cosine Similarity
        predict = l_cos_sim([source_output, dest_output])
        self._cos_model = Model(
            name=self._name,
            inputs={'org': source_input, 'obf': dest_input},
            outputs=predict
        )

        # compile model
        self._cos_model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        self._history = None
