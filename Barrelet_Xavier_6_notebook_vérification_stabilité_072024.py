import json
import os
import shutil
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from stackapi import StackAPI

warnings.filterwarnings("ignore", category=DeprecationWarning)

# STACKAPI CONFIGURATION
SITE = StackAPI('stackoverflow')

# MISC CONFIGURATION
plt.style.use("fivethirtyeight")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# PATHS
RESULTS_PATH = 'stability_results'
MODEL_PATH = 'inferring_api/models_XGB_45k/best_supervised_model.model'
TAGS_PATH = 'inferring_api/models_XGB_45k/tags.json'


def fetch_questions_from_month(month: int):
    """Fetches the questions of a given month with at least 100 votes."""
    # https://stackapi.readthedocs.io/en/latest/user/complex.html
    questions = SITE.fetch('questions',
                           fromdate=datetime(2023, month, 1),
                           todate=datetime(2023, month, 31),
                           min=100,
                           sort='votes',
                           filter='withbody'
                           )

    extracted_questions = questions['items']
    return [{
        "body": question['body'],
        "score": question['score'],
        "tags": question['tags'],
        "title": question['title']
    } for question in extracted_questions]


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


if __name__ == '__main__':
    print("Starting stability verification script.\n")
    remove_last_generated_results()


    # En attendant leur mise œuvre ultérieure, vous avez prévu de vérifier la stabilité du modèle dans le temps
    # sur 1 an, en mesurant mensuellement l’évolution des mesures et scores des questions de chacun de ces mois.

    count_cached_questions_per_year()

    # TODO: A simple predict with an accuracy score could be enough here, or a jaccard and hammond as well? What metric can you use? What threshold to use?
    # I'm not sure Frouros will be useful as you cannot compare the single feature of the dataset

    for month in range(1, 2):
        pass
    # for month in range(1, 13):
    #     monthly_questions = fetch_questions_from_month(month)


    print("\nStability verification script finished.\n")
