import json
import warnings
from datetime import datetime
from pprint import pprint

import pandas as pd
from pandas import DataFrame
from stackapi import StackAPI

warnings.filterwarnings("ignore", category=DeprecationWarning)

# STACKAPI CONFIGURATION
SITE = StackAPI('stackoverflow')
SITE.page_size = 50
SITE.max_pages = 1

# PANDAS CONFIGURATION
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def fetch_questions():
    # https://stackapi.readthedocs.io/en/latest/user/complex.html
    questions = SITE.fetch('questions',
                           fromdate=datetime(2020, 1, 1),
                           todate=datetime(2024, 7, 30),
                           min=50,
                           sort='votes',
                           filter='withbody',
                           # tagged='python'
                           )
    return questions


if __name__ == '__main__':
    print("Starting API script.\n")

    json_questions_answer = fetch_questions()
    extracted_questions = json_questions_answer['items']

    questions = [{
        "body": question['body'],
        "creation_date": question['creation_date'],
        "score": question['score'],
        "tags": question['tags'],
        "title": question['title']
    } for question in extracted_questions]

    print(f"{len(questions)} questions received from API.\n")

    df: DataFrame = DataFrame(questions)

    print("Displaying first line of dataframe:\n")
    pprint(df.head(1))

    print("\nAPI script now done.")
