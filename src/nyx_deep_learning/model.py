import json
from typing import Dict, List, Tuple
from pathlib import Path

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import pandas as pd

from loguru import logger


class NLPPipeline:

    def __init__(self, description_dataset: pd.DataFrame):
        self.description_dataset = description_dataset

    @property
    def preprocessor(self):
        try:
            return self._preprocessor
        except AttributeError:
            logger.info("Pulling preprocessor from TF Hub")
            self._preprocessor = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1')
            return self._preprocessor

    @property
    def encoder(self):
        try:
            return self._encoder
        except AttributeError:
            logger.info("Pulling encoder from TFHub")
            self._encoder = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3')
            return self._encoder

    @property
    def bert_model(self):
        try:
            return self._bert_model
        except AttributeError:
            logger.info("Pulling bert model from TFHub")
            self._bert_model = hub.load('https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1')
            return self._bert_model

    def preprocess(self) -> Tuple[List]:
        logger.info("Start preprocessing")

        # Train test splitting
        # Randomly sample data
        n_samples = len(self.description_dataset)
        train_idx = np.random.choice(range(n_samples), size=(int(0.8 * n_samples),), replace=False)
        test_idx =
        train_x = self.description_dataset.loc[train_idx]
        test_x = np.delete(self.description_dataset, train_idx)

        # Subset y
        mask = self.counts_dataset.index.isin(self.book_ids[train_idx])
        train_y = self.counts_dataset[mask]
        test_y = self.counts_dataset[~mask]
        breakpoint()

        assert len(train_y) == len(train_x)
        assert len(test_y) == len(test_x)

        return train_x, test_x, train_y, test_y

    def build_model(self):
        # From https://www.tensorflow.org/tutorials/text/classify_text_with_bert
        logger.info("Building model")
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(self.preprocessor, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(self.encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(1, activation='linear', name='regressor')(net)
        return tf.keras.Model(text_input, net)

    def fit_model(self, train_x, train_y, test_x, test_y, model):
        logger.info("Fitting model")
        model.compile(optimizer='SGD', loss='mse', metrics='mse')
        es = tf.keras.callbacks.EarlyStopping(patience=3)
        model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y), callbacks=es, epochs=100)
        return model

    def save_model(self, model, output_path):
        dest_path = Path(output_path) / 'model.h5'
        logger.info(f"Saving to {dest_path}")
        model.save(dest_path)

    def run(self):

        train_processed, test_processed, train_y, test_y = self.preprocess()
        model = self.build_model()
        model_fit = self.fit_model(train_processed, train_y, test_processed, test_y, model)
        self.save_model(model_fit, '/users/shawd/nyx/results')


def load_description_data(input_filepath: str, subsample: int = None) -> pd.DataFrame:

    df = pd.read_csv(input_filepath, use_cols=['text_reviews_count_bks', 'description'], subsample=subsample)
    return df

if __name__ == '__main__':

    description_dataset = load_description_data('/users/shawd/nyx/data/BooksMerged2000.csv')

    pipeline = NLPPipeline(description_dataset)
    pipeline.run()
