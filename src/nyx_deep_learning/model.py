import json
import requests
from typing import Dict, List, Tuple

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from loguru import logger


class NLPPipeline:

    def __init__(self, description_dataset: List, counts_dataset: pd.Series):
        self.description_dataset = description_dataset
        self.counts_dataset = counts_dataset

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
        ids = []
        descriptions = []
        for book in self.description_dataset:
            if book['description']:
                try:
                    ids.append(int(book['book_id']))
                except KeyError:
                    # For predictions, id might not be needed
                    logger.warning("Book_id not present in keys, omitting")
                descriptions.append(book['description'])

        # Train test splitting
        # Randomly sample data
        ids, descriptions = np.array(ids), np.array(descriptions)
        n_samples = len(descriptions)
        train_idx = np.random.choice(range(n_samples), size=(int(0.8 * n_samples),), replace=False)

        train_x = descriptions[train_idx]
        test_x = np.delete(descriptions, train_idx)

        # Subset y
        # mask = self.counts_dataset.index.isin(ids)
        # train_y = np.array(self.counts_dataset[mask])
        # test_y = np.array(self.counts_dataset[~mask])

        # HACK!!! Please remove. Only for testing
        train_y = np.random.randint(0, 20, (len(train_x),))
        test_y = np.random.randint(0, 20, (len(test_x),))

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

    def run(self):

        train_processed, test_processed, train_y, test_y = self.preprocess()
        model = self.build_model()
        model_fit = self.fit_model(train_processed, train_y, test_processed, test_y, model)
        self.save_model(model_fit)


def load_description_data(input_filepath: str, subsample: int = None):
    # TODO: Grab the Y variables!! From the other json, containing interactions
    with open(input_filepath, 'r') as f:
        if subsample:
            output = []
            for _ in range(subsample):
                book = json.loads(f.readline())
                output.append(book)
            return output
        else:
            data = json.load(f)
            return data


def load_book_counts_data(input_filepath: str, subsample: int = None) -> pd.Series:
    df = pd.read_csv(input_filepath, nrows=subsample)
    summary = df.groupby('book_id').is_read.sum()
    return summary


if __name__ == '__main__':

    description_dataset = load_description_data('/users/shawd/nyx/data/goodreads_books.json', subsample=100)

    book_counts_dataset = load_book_counts_data('/users/shawd/nyx/data/goodreads_interactions.csv', subsample=100)

    pipeline = NLPPipeline(description_dataset, book_counts_dataset)
    pipeline.run()
