import json
from datetime import datetime

from pandas import DataFrame
from stackapi import StackAPI

SITE = StackAPI('stackoverflow')
SITE.page_size = 50
SITE.max_pages = 1


def load_cached_questions():
    with open('questions.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        return json_data['items']


def cache_questions():
    # https://stackapi.readthedocs.io/en/latest/user/complex.html
    questions = SITE.fetch('questions',
                           fromdate=datetime(2022, 1, 1),
                           todate=datetime(2024, 7, 25),
                           min=50,
                           sort='votes',
                           tagged='python')

    with open('questions.json', 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    print("Starting project 5!\n")

    # cache_questions()

    json_questions = load_cached_questions()
    questions = [{
        "creation_date": question['creation_date'],
        "tags": question['tags'],
        "title": question['title'],
        "score": question['score'],

    } for question in json_questions]

    df: DataFrame = DataFrame(questions)
