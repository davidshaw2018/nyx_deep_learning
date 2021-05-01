import json
import requests
from typing import Dict, List, Tuple

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from sklearn.model_selection import train_test_split
import numpy as np

from loguru import logger


class NLPPipeline:

    def __init__(self, dataset: List):
        self.dataset = dataset

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

    def preprocess(self) -> Tuple[List]:
        logger.info("Start preprocessing")
        ids = []
        descriptions = []
        for book in self.dataset:
            if book['description']:
                try:
                    ids.append(int(book['book_id']))
                except KeyError:
                    # For predictions, id might not be needed
                    logger.warning("Book_id not present in keys, omitting")
                descriptions.append(book['description'])

        # Train test splitting
        # Randomly sample data
        descriptions = np.array(descriptions)
        n_samples = len(descriptions)
        train_idx = np.random.choice(range(n_samples), size=(0.8 * n_samples,))
        test_idx =


        # Tokenize the descriptions
        text_preprocessed = self.preprocessor(descriptions)



        return ids, text_preprocessed


def load_data(input_filepath: str, subsample: int = None):
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


if __name__ == '__main__':

    dataset = load_data('/users/shawd/nyx/data/goodreads_books.json', subsample=100)

    pipeline = NLPPipeline(dataset)
    ids, texts = pipeline.preprocess()
    breakpoint()
