import json

import joblib
import pandas as pd
from flask import Flask, jsonify, request
import tensorflow_hub as hub
from sklearn.preprocessing import MultiLabelBinarizer

# model = joblib.load('models/supervised/best_supervised_model.model')
model = joblib.load('best_supervised_model.model')
print("Model loaded from file.\n")

# with open('models/supervised/tags.json') as json_data:
with open('tags.json') as json_data:
    json_tags = json.load(json_data)
    tags_df = pd.Series(json_tags)
print("Tags loaded from file.\n")

multi_label_binarizer = MultiLabelBinarizer()
multi_label_binarizer.fit(tags_df)

app = Flask(__name__)
print("Flask app started.\n")


def transform_text(body, title):
    sentence = [f"{body} {title}"]
    us_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    return us_encoder(sentence)


@app.route("/")
def health_check():
    return "Inferring service up!"


@app.route("/predict", methods=['POST'])
def predict_text():
    body = request.json.get("body")
    title = request.json.get("title")
    print(f"body:{body}, title:{title}")

    text = transform_text(body, title)

    prediction = model.predict(text)

    tags = multi_label_binarizer.inverse_transform(prediction)
    print(f"prediction:{tags}")

    return jsonify({"predicted_tags": tags})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
