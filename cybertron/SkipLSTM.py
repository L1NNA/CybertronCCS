import sys, os
from os.path import isfile, join

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.python.keras.layers.recurrent import _generate_zero_filled_state_for_cell

from cybertron.BaseModel import BaseModel


'''
Modern Tensorflow Implementation of skip-LSTM: https://arxiv.org/abs/1708.06834
'''
class SkipLSTMLayer(Layer):
    '''
    num_cells: number of nodes in the layer
    name: name for the layer
    '''
    def __init__(self, num_cells=128, name='SkipLSTMLayer', **kwargs):
        super(SkipLSTMLayer, self).__init__(name=name, **kwargs)
        self.num_cells = num_cells
        self.rnn_cell = LSTMCell(self.num_cells)
        self.dense = Dense(1, use_bias=True, activation='sigmoid')
        
    # Build the basic cells. Done automatically for dense layer
    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if not self.rnn_cell.built:
            with K.name_scope(self.rnn_cell.name):
                self.rnn_cell.build(input_shape)
                self.rnn_cell.built = True
    
    # Used when this layer is part of a network
    # o_inputs is of shape [batch, window size, embedding_dim]
    def call(self, o_inputs, **kwargs):
        # o_inputs is of shape [batch, window size, embedding_dim]
        win_size = tf.shape(o_inputs)[1]
        batch_size = tf.shape(o_inputs)[0]
        
        results = tf.TensorArray(dtype=tf.float32, size=win_size)
        
        '''
        This function is applied to each timestamp of the sequence for the node
        Inputs/Outputs:
            t: current timestamp
            ut: state update gate for current timestamp
            state:  array containing: [ht_1, ct_1]: hidden and candidates from previous timestamp
            res: stacking array containing node output values
        '''
        def _step(t, state, ut, res):
            
            # determine the state of the update gate: {0,1} (Eq 2)
            ut_gate = tf.round(ut)
            
            # generate output for current timestamp t (Eq 3)
            (out, state_n) = self.rnn_cell(o_inputs[:, t, :], state)

            # determine whether to actually update the state based on update gate (Eq 3)
            ht_n = tf.expand_dims(ut_gate, 1) * state[0] + tf.expand_dims(1 - ut_gate, 1) * state_n[0]
            ct_n = tf.expand_dims(ut_gate, 1) * state[1] + tf.expand_dims(1 - ut_gate, 1) * state_n[1]
            state_n = [ht_n, ct_n]
            
            # compute the change in update gate value based on hidden state (Eq 4)
            # concatenate the hidden and candidate so it can be passed
            delta_ut = tf.squeeze(self.dense(tf.concat([ht_n, ct_n], 1)), 1)
            
            # compute the value of the update gate for the next timestamp (Eq 5)
            ut_n = ut_gate * delta_ut + (1 - ut_gate) * (ut + tf.minimum(delta_ut, 1-ut))
            
            # write node output to index t of array res (returned the updated res array)
            res_updated = res.write(t, out)
            
            return t+1, state_n, ut_n, res_updated
        
        # generate initial weights of all 0
        state0 = _generate_zero_filled_state_for_cell(self.rnn_cell, o_inputs, None, None)
        u0 = tf.ones(batch_size, dtype=tf.float32)
        
        '''
        Loop through timestamps and return the pile of node outputs
        Inputs:
            Stop condition on while loop
            Step function applied at each iteration
            Initial values in loop
        '''
        *_, final_res = tf.while_loop(
            lambda t, *_: t < win_size, 
            _step,
            (0, state0, u0, results)
        )
        
        final_res = final_res.stack()
        # [time, batch, cell_dim]
        final_res = tf.transpose(final_res, [1, 0, 2])
        # [batch, time, cell_dim]
        return final_res


class SkipLSTM(BaseModel):
    def __init__(self, max_seq_len, name=None):
        super(SkipLSTM, self).__init__(
            max_seq_len=max_seq_len, name=name if name else 'SkipLSTM')

    def build(self):
        input_dim = len(self.vectorizer.get_vocabulary())
        input_length = self.max_seq_len

        # Build source model
        source_input = Input(shape=(input_length,))
        dest_input = Input(shape=(input_length,))
        embedding_layer = Embedding(
            input_dim=input_dim, output_dim=128, input_length=input_length)

        h3 = SkipLSTMLayer(128, name='h2')
        h4 = SkipLSTMLayer(128, name='h3')
        dense_layer = Dense(128)

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
        cos_sim = Dot(axes=1, normalize=True)([source_output, dest_output])
        cos_sim = (cos_sim + 1) / 2
        predict = tf.clip_by_value(cos_sim, clip_value_min=0, clip_value_max=1)
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
