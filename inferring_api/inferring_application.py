import logging
import os
from io import BytesIO

import joblib
import pandas as pd
import psycopg
import requests
import tensorflow_hub as hub
from flask import Flask, jsonify, request, current_app
from sklearn.preprocessing import MultiLabelBinarizer

DEFAULT_MODEL_URL = "https://inferring-api-models.s3.eu-west-3.amazonaws.com/best_supervised_model.model"
DEFAULT_TAGS_URL = "https://inferring-api-models.s3.eu-west-3.amazonaws.com/tags.json"


def load_model(app):
    model_url = os.getenv("MODEL_URL", default=DEFAULT_MODEL_URL)

    file = requests.get(model_url).content
    app.model = joblib.load(BytesIO(file))

    app.logger.info(f"Model loaded from url:{model_url}.\n")


def start_label_binarizer(app):
    tags_url_from_env_var = os.getenv("TAGS_URL", default=DEFAULT_TAGS_URL)

    json_tags = requests.get(tags_url_from_env_var).json()
    app.logger.info(f"Tags loaded from url:{tags_url_from_env_var}.\n")

    tags_df = pd.Series(json_tags)
    app.multi_label_binarizer = MultiLabelBinarizer()
    app.multi_label_binarizer.fit(tags_df)


def init_db(app):
    connection_url = f'postgresql://{os.getenv("RDS_USERNAME")}:{os.getenv("RDS_PASSWORD")}@{os.getenv("RDS_HOSTNAME")}:{os.getenv("RDS_PORT")}/{os.getenv("RDS_DB_NAME")}'
    app.db_connection = psycopg.connect(connection_url, autocommit=True)

    with app.db_connection.cursor() as cursor:
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS events (event_id SERIAL PRIMARY KEY, body TEXT, title TEXT, tags TEXT);""")
    app.logger.info("Db connection initialized, table events created.\n")


def create_app():
    app = Flask(__name__)
    app.logger.setLevel(logging.INFO)

    load_model(app)
    start_label_binarizer(app)
    init_db(app)

    app.us_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    app.logger.info("Universal sentence encoder loaded.\n")

    def transform_text(body, title):
        sentence = [f"{body} {title}"]
        return app.us_encoder(sentence)

    @app.route("/")
    def health_check():
        return "Inferring service up!"

    @app.route("/predict", methods=['POST'])
    def predict_text():
        body = request.json.get("body")
        title = request.json.get("title")

        text = transform_text(body, title)

        prediction = app.model.predict(text)
        tags = app.multi_label_binarizer.inverse_transform(prediction)[0]

        with app.db_connection.cursor() as cursor:
            cursor.execute("""INSERT INTO events (body, title, tags) VALUES (%s, %s, %s);""", (body, title, tags))

        current_app.logger.info(f"Predicted tags:{tags} for title:{title} and body:{body}.\n")

        return jsonify({"predicted_tags": tags})

    @app.route("/events")
    def get_events():
        with app.db_connection.cursor() as cursor:
            cursor.execute("""SELECT * FROM events;""")
            events = cursor.fetchall()

        return jsonify(events)

    app.logger.info("Flask webapp is up and running.\n")
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=8000)
