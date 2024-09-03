import json
import os
import shutil
import warnings
from datetime import datetime
from os.path import exists

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow_hub as hub
from pandas import DataFrame
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from stackapi import StackAPI

warnings.filterwarnings("ignore", category=DeprecationWarning)

# STACKAPI CONFIGURATION
SITE = StackAPI('stackoverflow')
SITE.max_pages = 25

# MISC CONFIGURATION
plt.style.use("fivethirtyeight")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# PATHS
CACHED_2023_QUESTIONS_FILE = 'cached_questions_2023.json'
RESULTS_PATH = 'stability_results'
MODEL_PATH = 'inferring_api/models_XGB_45k/best_supervised_model.model'
TAGS_PATH = 'inferring_api/models_XGB_45k/tags.json'

# USE ENCODER
use_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def cache_questions_from_2023():
    """Fetches the questions of a given month with at least 20 votes."""
    # https://stackapi.readthedocs.io/en/latest/user/complex.html
    questions = SITE.fetch('questions',
                           fromdate=datetime(2023, 1, 1),
                           todate=datetime(2023, 12, 31),
                           min=10,
                           sort='votes',
                           filter='withbody'
                           )

    extracted_questions = questions['items']
    questions = [{
        "body": question['body'],
        "creation_date": question['creation_date'],
        "score": question['score'],
        "tags": question['tags'],
        "title": question['title']
    } for question in extracted_questions]

    with open(CACHED_2023_QUESTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)


def remove_last_generated_results():
    """Removes the content of the saved plots."""
    shutil.rmtree(RESULTS_PATH, ignore_errors=True)
    os.mkdir(RESULTS_PATH)


def add_datetime_to_question(question: dict):
    """Add a datetime field to the question."""
    question['creation_datetime'] = datetime.fromtimestamp(question['creation_date'])
    return question


def count_cached_questions_per_year():
    """Displays each year's number of cached questions."""
    with open('cached_questions.json', 'r', encoding='utf-8') as f:
        questions = json.load(f)

        min_timestamp = min([question['creation_date'] for question in questions])
        min_datetime = datetime.fromtimestamp(min_timestamp)

        max_timestamp = max([question['creation_date'] for question in questions])
        max_datetime = datetime.fromtimestamp(max_timestamp)

        print(f"The cached questions range from {min_datetime} to {max_datetime}.\n")

        enriched_questions = list(map(add_datetime_to_question, questions))
        for year in range(2010, 2024):
            number_of_questions_for_year = len(
                [question for question in enriched_questions if question['creation_datetime'].year == year])
            print(f"Number of questions for year {year}: {number_of_questions_for_year}")


def count_cached_questions_per_month(questions):
    """Displays each year's number of cached questions."""

    min_timestamp = min([question['creation_date'] for question in questions])
    min_datetime = datetime.fromtimestamp(min_timestamp)

    max_timestamp = max([question['creation_date'] for question in questions])
    max_datetime = datetime.fromtimestamp(max_timestamp)

    print(f"The cached questions range from {min_datetime} to {max_datetime}.\n")

    for month in range(1, 13):
        number_of_questions_for_month = len([question for question in questions
                                             if question['creation_datetime'].month == month])
        print(f"Number of questions for month {month}: {number_of_questions_for_month}")


def load_cached_2023_questions():
    """Load questions from the cache file."""
    with open(CACHED_2023_QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def start_label_binarizer():
    json_tags = json.load(open(TAGS_PATH, 'r'))

    tags_df = pd.Series(json_tags)
    multi_label_binarizer = MultiLabelBinarizer()
    multi_label_binarizer.fit(tags_df)

    return multi_label_binarizer


def transform_text_using_USE(sentences):
    """Transform the text of the question's body and title into USE embeddings."""
    # We don't want to use the cleaned text field with USE, only title + " " + body
    sentences = [f"{sentence[1]} {sentence[0]}" for sentence in sentences.iterrows()]

    batch_size = 1
    for step in range(len(sentences) // batch_size):
        idx = step * batch_size
        feat = use_encoder(sentences[idx:idx + batch_size])

        if step == 0:
            features = feat
        else:
            features = np.concatenate((features, feat))

    return features


def create_results_plots(results):
    """Generate the plot showing the performances with each words embedding method for the Jaccard Score and Hamming Loss."""
    create_results_plot(results, "jaccard_score")
    create_results_plot(results, "hamming_loss")
    create_results_plot(results, "accuracy_score")
    create_results_plot(results, "f1_score")


def create_results_plot(results, metric):
    performance_plot = (results[[metric, "month"]]
                        .plot(kind="bar", x="month", figsize=(15, 8), rot=0,
                              title=f"Models Performance Sorted by {metric}"))
    performance_plot.title.set_size(20)
    performance_plot.set(xlabel=None)

    performance_plot.get_figure().savefig(f"{RESULTS_PATH}/performance_{metric}_plot.png", bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    print("Starting stability verification script.\n")
    remove_last_generated_results()

    if not exists(CACHED_2023_QUESTIONS_FILE):
        print(f"Cached questions are missing, downloading them in {CACHED_2023_QUESTIONS_FILE}.\n")
        cache_questions_from_2023()

    cached_questions_2023 = load_cached_2023_questions()
    print(f"\nNumber of questions of 2023 with a minimum of 10 votes: {len(cached_questions_2023)}.\n")
    cached_questions_2023 = list(map(add_datetime_to_question, cached_questions_2023))

    model = joblib.load(MODEL_PATH)
    label_binarizer = start_label_binarizer()

    results = []
    for month in range(1, 13):
        current_month_questions = DataFrame([question for question in cached_questions_2023
                                             if question['creation_datetime'].month == month])
        print(f"Starting month:{month} verification with {len(current_month_questions)} questions.")

        y_test = label_binarizer.transform(current_month_questions['tags'])

        questions_without_tags = current_month_questions.drop(columns=['tags'], axis=1)
        x_test = transform_text_using_USE(questions_without_tags)

        predictions_test_y = model.predict(x_test)

        hamming_loss = metrics.hamming_loss(y_true=y_test, y_pred=predictions_test_y)
        jaccard_score = metrics.jaccard_score(y_true=y_test, y_pred=predictions_test_y, average='samples',
                                              zero_division=0)
        accuracy_score = metrics.accuracy_score(y_true=y_test, y_pred=predictions_test_y)
        f1_score = metrics.f1_score(y_true=y_test, y_pred=predictions_test_y, average='samples', zero_division=0)

        results.append({
            "month": month,
            "hamming_loss": hamming_loss,
            "jaccard_score": jaccard_score,
            "accuracy_score": accuracy_score,
            "f1_score": f1_score
        })

        print(f"Results of month:{month} - Hamming loss:{hamming_loss}, jaccard_score:{jaccard_score}, "
              f"accuracy_score:{accuracy_score}, f1_score:{f1_score}\n")

    create_results_plots(DataFrame(results))

    print("\nStability verification script finished.\n")
