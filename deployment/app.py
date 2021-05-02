from flask import Flask, request
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_text as text
from typing import List

app = Flask(__name__)
model = keras.models.load_model('model')


@app.route('/predict_sales', method=['POST'])
def predict_sales():
    # Unpack the json data string, and get a pred
    tagline: List[str] = request.form['tagline']

    # Predict
    expected_reads: List[int] = model.predict(tagline)
    return expected_reads

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0')
