import os

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from cybertron.BaseModel import BaseModel


def cos_sim(input_length, input_dim):
    # layers
    l_source_input = Input(shape=(input_length, input_dim))
    l_dest_input = Input(shape=(input_length, input_dim))
    l_rnn = Bidirectional(GRU(128, return_sequences=True))
    l_rnn_2 = Bidirectional(GRU(128))
    l_dense = Dense(128)
    l_cos_sim = Dot(axes=1, normalize=True)

    # output
    source_context = l_dense(l_rnn_2(l_rnn(l_source_input)))
    dest_context = l_dense(l_rnn_2(l_rnn(l_dest_input)))
    predict = l_cos_sim([source_context, dest_context])
    predict = tf.clip_by_value((predict + 1) / 2, 0, 1)

    return Model([l_source_input, l_dest_input], predict, name='cos_sim')


def abstraction(l_input, best_index, input_length, filters, new_len, input_dim):
    # select top tokens with number of new_len
    best_index = tf.clip_by_value(
        best_index,
        clip_value_max=tf.cast(max(input_length - new_len, 0), 'int64'),
        clip_value_min=tf.cast(0, 'int64')
    )
    offset = tf.tile(tf.reshape(tf.range(0, new_len, dtype='int64'), [1, new_len]), [filters, 1])
    clipped = tf.clip_by_value(tf.expand_dims(best_index, -1) + tf.expand_dims(offset, 0),
                               clip_value_max=tf.cast(input_length - 1, 'int64'),
                               clip_value_min=tf.cast(0, 'int64'))
    selected = tf.gather(l_input, clipped, batch_dims=1)  # batch x filters x new_len x input_dim
    selected = tf.reshape(selected, [-1, filters * new_len, input_dim])
    return selected


def selection(input_length, input_dim, filters, new_len, model_path=None, kernel_size=10):
    # layers
    l_input = Input(shape=(input_length, input_dim))
    l_prob = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, use_bias=True, padding='same') # same is necessary
    
    l_softmax = Lambda(lambda x : tf.math.softmax(x, axis=1), name='softmax')
    l_reduce_max = Lambda(lambda x : tf.reduce_max(x, axis=1), name='reduce_max')
    l_argmax = Lambda(lambda x : tf.math.argmax(x, axis=1), name='argmax')
    
    if model_path != None:
        probs_txt = os.path.join(model_path, 'probs.txt')
        probs_txt = os.path.abspath(probs_txt)
        probs_txt = 'file://' + probs_txt
        def indices_track(indices):
            tf.print(indices[0], output_stream=probs_txt)
            return indices
        l_recording = Lambda(indices_track, name='recording')
    
    l_abstraction = Lambda(lambda x : abstraction(x[0], x[1], input_length, filters, new_len, input_dim), name='abstraction')
    
    # calculate probabilities
    logits = l_softmax(l_prob(l_input))
    probs = l_reduce_max(logits)

    # select top tokens with number of new_len
    best_indices = l_argmax(logits)
    if model_path != None:
        best_indices = l_recording(best_indices)
    selected = l_abstraction([l_input, best_indices])

    return Model(l_input, [selected, probs], name='selection')


class MaximumLikelihoodLoss(tf.keras.losses.Loss):
    
    def __init__(self, filters, name='maximum_likelihood_loss'):
        super().__init__(name=name)
        self.filters = filters

    def call(self, y_true, y_pred):
        y_true = tf.slice(y_true, [0, 0], [-1, 1])
        predict = tf.slice(y_pred, [0, 0], [-1, 1])
        probs = tf.slice(y_pred, [0, 1], [-1, self.filters])
        
        cond = tf.cast(tf.round(predict), 'int64') == y_true
        rewards = tf.where(cond, tf.ones(tf.shape(predict)), -tf.ones(tf.shape(predict)))

        loss = -1 * tf.math.log(probs + 1e-10) * rewards
        loss = tf.reduce_sum(loss, axis=1)
        return tf.reduce_mean(loss)


class RLModel(BaseModel):

    def __init__(self, max_seq_len=2048, embed_dim=128, filters=3,
                 output_length=128, name=None, model_path=None):
        super(RLModel, self).__init__(max_seq_len=max_seq_len,
                                      name=name if name else 'RL')
        self.embed_dim = embed_dim
        self.filters = filters
        self.output_length = output_length
        self.model_path = model_path
        self.early_stopping = False
        self._abstraction_model = None
        self._cos_sim_model = None

    def preprocess(self, pairs: tf.data.Dataset, train=False, batch_size=64):
        pairs = super().preprocess(pairs, train, batch_size)
        pairs = pairs.map(lambda x, y: (x, (y, tf.repeat(y, self.filters + 1, axis=1))))
        return pairs

    def build(self):
        input_dim = len(self.vectorizer.get_vocabulary())
        input_length = self.max_seq_len
        # input and embedding
        l_source_input = Input(shape=(input_length,))
        l_dest_input = Input(shape=(input_length,))
        l_embedding = Embedding(
            input_dim=input_dim,
            output_dim=self.embed_dim,
            input_length=input_length
        )
        # text selection layer
        l_abstraction = selection(
            input_length,
            self.embed_dim,
            self.filters,
            self.output_length,
            self.model_path,
            kernel_size=5
        )
        # comparison layer
        l_cos_sim = cos_sim(self.output_length * self.filters, self.embed_dim)
        # lambdas
        l_probs_mean = Lambda(lambda x : (x[0] + x[1]) / 2, name='probs_mean')
        l_stack_loss = Lambda(lambda x : tf.concat([x[0], x[1]], axis=1), name='mll')
        

        # connect together
        source_selected, source_prob = l_abstraction(l_embedding(l_source_input))
        dest_selected, dest_prob = l_abstraction(l_embedding(l_dest_input))
        predict = l_cos_sim([source_selected, dest_selected])
        prob = l_probs_mean([source_prob, dest_prob])
        maxlikelihood = l_stack_loss([predict, prob])

        self._cos_model = Model(
            name=self._name,
            inputs={'org': l_source_input, 'obf': l_dest_input},
            outputs=[predict, maxlikelihood]
        )

        # compile model
        self._cos_model.compile(
            loss=['binary_crossentropy', MaximumLikelihoodLoss(self.filters)],
            optimizer='adam',
            metrics=[['accuracy', 'AUC', 'Precision', 'Recall'], []],
            loss_weights=[0.2, 0.8]
        )
        self._abstraction_model = l_abstraction
        self._cos_sim_model = l_cos_sim
        self._history = None
        
    def model_summary(self):
        self._abstraction_model.summary()
        self._cos_sim_model.summary()
        self._cos_model.summary()
