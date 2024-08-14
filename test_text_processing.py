import json

from Barrelet_Xavier_1_notebook_exploration_072024 import extract_and_clean_text

TEST_CACHED_QUESTIONS_FILE = "cached_questions_100.json"


def load_test_questions():
    with open(TEST_CACHED_QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)



def test_extract_and_clean_text_removes_tags():
    # GIVEN
    test_question = {
        "title": "<i>Hello</i>",
        "body": """<ul>
        <li>Coffee</li>
        <li>Tea</li>
        <li>Milk</li>
        </ul>""",
    }

    # WHEN
    cleaned_text = extract_and_clean_text(test_question)['text']

    # THEN
    assert cleaned_text == 'hello coffee tea milk'


def test_extract_and_clean_text_removes_punctuation():
    # GIVEN
    test_question = {
        "title": "Jean,",
        "body": "like. dark! beer?",
    }

    # WHEN
    cleaned_text = extract_and_clean_text(test_question)['text']

    # THEN
    assert cleaned_text == 'jean like dark beer'


def test_extract_and_clean_text_removes_multiple_whitespaces():
    # GIVEN
    test_question = {
        "title": "Jean     ",
        "body": "like  dark    beer",
    }

    # WHEN
    cleaned_text = extract_and_clean_text(test_question)['text']

    # THEN
    assert cleaned_text == 'jean like dark beer'


def test_extract_and_clean_text_removes_numbers():
    # GIVEN
    test_question = {
        "title": "Jean",
        "body": "like 42 beer",
    }

    # WHEN
    cleaned_text = extract_and_clean_text(test_question)['text']

    # THEN
    assert cleaned_text == 'jean like beer'


def test_extract_and_clean_text_removes_stopwords():
    # GIVEN
    test_question = {
        "title": "A bear",
        "body": "like the honey from this tree",
    }

    # WHEN
    cleaned_text = extract_and_clean_text(test_question)['text']

    # THEN
    assert cleaned_text == 'bear like honey tree'


def test_extract_and_clean_text_removes_short_words():
    # GIVEN
    test_question = {
        "title": "We",
        "body": "like beer",
    }

    # WHEN
    cleaned_text = extract_and_clean_text(test_question)['text']

    # THEN
    assert cleaned_text == 'like beer'


def test_extract_and_clean_text_lower_text():
    # GIVEN
    test_question = {
        "title": "jeAN",
        "body": "lIKe BEEr",
    }

    # WHEN
    cleaned_text = extract_and_clean_text(test_question)['text']

    # THEN
    assert cleaned_text == 'jean like beer'


def test_extract_and_clean_text_lemmatize_words():
    # GIVEN
    test_question = {
        "title": "babies",
        "body": "is watching geese",
    }

    # WHEN
    cleaned_text = extract_and_clean_text(test_question)['text']

    # THEN
    assert cleaned_text == 'baby watching goose'
