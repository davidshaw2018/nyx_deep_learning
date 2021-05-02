from flask import Flask, request
import tensorflow_text as text  # noqa
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model('model')


@app.route('/predict_sales', method=['POST'])
def predict_sales():
    # Unpack the json data string, and get a pred
    tagline = request.form['tagline']

    # Predict
    expected_reads = model.predict(tagline)
    return expected_reads
