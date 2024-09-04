import os

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from cybertron.BaseModel import BaseModel


class GRUC(BaseModel):

    def __init__(self, max_seq_len=512, name=None):
        super(GRUC, self).__init__(max_seq_len=max_seq_len,
                         name=name if name else 'GRUC')

    def build(self):
        input_dim = len(self.vectorizer.get_vocabulary())
        input_length = self.max_seq_len
        
        # Build layers
        l_source_input = Input(shape=(input_length,))
        l_dest_input = Input(shape=(input_length,))
        l_embedding = Embedding(
            input_dim=input_dim,
            output_dim=128,
            input_length=input_length
        )
        l_h2 = Bidirectional(GRU(128, return_sequences=True))
        l_h3 = Bidirectional(GRU(128, return_sequences=False))
        l_dense = Dense(128)
        l_cos_sim = self._cos_sim(128)

        # Link layers
        source_output = l_dense(
            l_h3(l_h2(l_embedding(l_source_input)))
        )
        dest_output = l_dense(
            l_h3(l_h2(l_embedding(l_dest_input)))
        )

        # Build Cosine Similarity
        predict = l_cos_sim([source_output, dest_output])
        self._cos_model = Model(
            name=self._name,
            inputs={'org': l_source_input, 'obf': l_dest_input},
            outputs=predict
        )

        # compile model
        self._cos_model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        self.history = None
