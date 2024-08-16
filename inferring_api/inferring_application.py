import json

import joblib
import pandas as pd
from flask import Flask, jsonify, request, current_app
import tensorflow_hub as hub
from sklearn.preprocessing import MultiLabelBinarizer

app = Flask(__name__)
print("Flask app started.\n")


# app.model = joblib.load('models/best_supervised_model.model')
app.model = joblib.load('best_supervised_model.model')
print("Model loaded from file.\n")


# with open('models/tags.json') as json_data:
with open('tags.json') as json_data:
    json_tags = json.load(json_data)
    tags_df = pd.Series(json_tags)
print("Tags loaded from file.\n")

app.multi_label_binarizer = MultiLabelBinarizer()
app.multi_label_binarizer.fit(tags_df)



def transform_text(body, title):
    sentence = [f"{body} {title}"]
    us_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    return us_encoder(sentence)


@app.route("/")
def health_check():
    return "Inferring service up2!"


@app.route("/predict", methods=['POST'])
def predict_text():
    body = request.json.get("body")
    title = request.json.get("title")
    current_app.logger.info(f"Payload received:\ntitle:{title},\nbody:{body}")

    text = transform_text(body, title)

    prediction = app.model.predict(text)

    tags = app.multi_label_binarizer.inverse_transform(prediction)[0]
    current_app.logger.info(f"Tags predicted:{tags}")

    return jsonify({"predicted_tags": tags})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
