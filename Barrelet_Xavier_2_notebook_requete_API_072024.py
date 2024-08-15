import warnings
from datetime import datetime
from pprint import pprint

import pandas as pd
from pandas import DataFrame
from stackapi import StackAPI

warnings.filterwarnings("ignore", category=DeprecationWarning)

# STACKAPI CONFIGURATION. No need for an API key here as we only want to fetch 50 results.
SITE = StackAPI('stackoverflow')
SITE.page_size = 50
SITE.max_pages = 1

# PANDAS CONFIGURATION
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def fetch_questions():
    """Fetch 50 questions with at least 50 votes from the StackOverflow API."""
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

    trimmed_questions = [{
        "body": question['body'],
        "score": question['score'],
        "tags": question['tags'],
        "title": question['title']
    } for question in extracted_questions]

    print(f"{len(trimmed_questions)} questions received from API.\n")

    df: DataFrame = DataFrame(trimmed_questions)

    print("Displaying first ten lines of dataframe:\n")
    pprint(df.head(10))

    print("\nAPI script now done.")
