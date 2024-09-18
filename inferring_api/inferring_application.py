import logging
import os
from io import BytesIO

import gensim.parsing.preprocessing as gsp
import joblib
import nltk
import psycopg
import requests
from flask import Flask, jsonify, request
from nltk import WordNetLemmatizer
from psycopg.rows import dict_row

DEFAULT_MODEL_URL = "https://inferring-api-models.s3.eu-west-3.amazonaws.com/best_supervised_model.model"
DEFAULT_ML_BINARIZER_URL = "https://inferring-api-models.s3.eu-west-3.amazonaws.com/best_ml_binarizer.model"
DEFAULT_EMBEDDER_URL = "https://inferring-api-models.s3.eu-west-3.amazonaws.com/embedder_model.model"

nltk.download('wordnet')
nltk.download('punkt')


def load_model(app):
    model_url = os.getenv("MODEL_URL", default=DEFAULT_MODEL_URL)

    file = requests.get(model_url).content
    app.model = joblib.load(BytesIO(file))

    app.logger.info(f"Model loaded from url:{model_url}.\n")


def load_label_binarizer(app):
    binarizer_url = os.getenv("ML_BINARIZER_URL", default=DEFAULT_ML_BINARIZER_URL)

    file = requests.get(binarizer_url).content
    app.binarizer = joblib.load(BytesIO(file))

    app.logger.info(f"Multilabel binarizer loaded from url:{binarizer_url}.\n")


def load_embedder(app):
    embedder_url = os.getenv("EMBEDDER_URL", default=DEFAULT_EMBEDDER_URL)

    file = requests.get(embedder_url).content
    app.embedder = joblib.load(BytesIO(file))

    app.logger.info(f"Words embedder loaded from url:{embedder_url}.\n")


def init_db(app):
    connection_url = f'postgresql://{os.getenv("RDS_USERNAME")}:{os.getenv("RDS_PASSWORD")}@{os.getenv("RDS_HOSTNAME")}:{os.getenv("RDS_PORT")}/{os.getenv("RDS_DB_NAME")}'
    app.db_connection = psycopg.connect(connection_url, autocommit=True, row_factory=dict_row)

    with app.db_connection.cursor() as cursor:
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS events (event_id SERIAL PRIMARY KEY, body TEXT, title TEXT, tags TEXT);""")
    app.logger.info("Db connection initialized, table events created.\n")


def extract_and_clean_text(title: str, body: str, lemmatizer):
    """Create a new 'text' field for each question containing the cleaned, tokenized and lemmatized title + body."""
    text = f"{title} {body}"

    for filter in [gsp.strip_tags,
                   gsp.strip_punctuation,
                   gsp.strip_multiple_whitespaces,
                   gsp.strip_numeric,
                   gsp.remove_stopwords,
                   gsp.strip_short,
                   gsp.lower_to_unicode]:
        text = filter(text)

    cleaned_text = text.replace("quot", "")
    tokenized_text = nltk.tokenize.word_tokenize(cleaned_text)

    words_lemmatized = [lemmatizer.lemmatize(w) for w in tokenized_text]
    return " ".join(words_lemmatized)


def create_app():
    app = Flask(__name__)
    app.logger.setLevel(logging.INFO)

    load_model(app)
    load_label_binarizer(app)
    load_embedder(app)
    init_db(app)

    lemmatizer = WordNetLemmatizer()

    @app.route("/")
    def health_check():
        return "Inferring service up!"

    @app.route("/predict", methods=['POST'])
    def predict_text():
        body = request.json.get("body")
        title = request.json.get("title")

        text = extract_and_clean_text(title, body, lemmatizer)
        transformed_text = app.embedder.transform([text])

        prediction = app.model.predict(transformed_text)
        tags = app.binarizer.inverse_transform(prediction)[0]

        app.db_connection.execute("INSERT INTO events (body, title, tags) VALUES (%s, %s, %s);", (body, title, tags))

        app.logger.info(f"Predicted tags:{tags} for title:{title} and body:{body}.\n")
        return jsonify({"predicted_tags": tags})

    @app.route("/events")
    def get_events():
        with app.db_connection.cursor() as cursor:
            cursor.execute("SELECT * FROM events;")
            events = cursor.fetchall()

        return jsonify(events)

    app.logger.info("Flask webapp is up and running.\n")
    return app


def create_gunicorn_app(environ, start_response):
    return create_app()

if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=8000)
