from flask import Flask, request
import tensorflow_hub as hub
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model('model.h5')
tokenizer = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1')


@app.route('/predict_sales', method=['POST'])
def predict_sales():
    # Unpack the json data string, and get a pred
    tagline = request.form['tagline']

    # Tokenize
    tokenized = tokenizer(tagline)

    # Predict
    expected_reads = model.predict(tokenized)
    return expected_reads
