import json
import os
import shutil
import string
import warnings
from pprint import pprint

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk import WordNetLemmatizer
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.multioutput import MultiOutputClassifier
import gensim.parsing.preprocessing as gsp
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore", category=DeprecationWarning)

# MISC CONFIGURATION
plt.style.use("fivethirtyeight")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

RESULTS_PATH = 'supervised_results'

# NLTK PACKAGES
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('words')
# nltk.download('wordnet')

stopwords = nltk.corpus.stopwords.words('english')
words = set(nltk.corpus.words.words())
lemmatizer = WordNetLemmatizer()


def load_cached_questions():
    """Load questions from the cache file."""
    with open('cached_questions.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        return json_data['items']


def remove_last_generated_results():
    """Removes the content of the saved plots."""
    shutil.rmtree(RESULTS_PATH, ignore_errors=True)
    os.mkdir(RESULTS_PATH)


def extract_and_clean_text(question: dict):
    title = question['title']
    body = question['body']
    text = f"{title} {body}"

    text_without_punctuation = "".join([i.lower() for i in text if i not in string.punctuation])
    text_without_number = ''.join(i for i in text_without_punctuation if not i.isdigit())

    tokenized_text = nltk.tokenize.word_tokenize(text_without_number)
    # Words with low information amount such as the, a, an, etc.
    words_without_stopwords = [i for i in tokenized_text if i not in stopwords]

    words_without_tags = (gsp.strip_tags(word) for word in words_without_stopwords)
    words_without_short_words = (gsp.strip_short(word) for word in words_without_tags)

    # Keeping only the common part of verbs for example
    words_lemmatized = (lemmatizer.lemmatize(w) for w in words_without_short_words)
    cleaned_text = ' '.join(w for w in words_lemmatized if w in words or not w.isalpha())
    question['text'] = cleaned_text

    bigrams = nltk.bigrams(tokenized_text)
    question['bigrams'] = [' '.join(bigram) for bigram in bigrams]

    trigrams = nltk.trigrams(tokenized_text)
    question['trigrams'] = [' '.join(trigram) for trigram in trigrams]

    return question


def perform_supervised_modeling(questions):
    questions_df = DataFrame(questions)

    tags = MultiLabelBinarizer().fit_transform(questions_df['tags'])
    questions_without_tags = questions_df.drop(columns=['tags'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(questions_without_tags, tags, test_size=0.2,
                                                        random_state=42)
    pprint(y_train)

    print(f"training set size:{len(x_train)}, test set size:{len(x_test)}\n")

    rf_hyperparameters = {'estimator__max_depth': range(2, 8), 'estimator__max_features': range(2, 10)}

    # https://pub.towardsai.net/understanding-multi-label-classification-model-and-accuracy-metrics-1b2a8e2648ca
    # https://pub.towardsai.net/multi-label-text-classification-using-scikit-multilearn-case-study-with-stackoverflow-questions-768cb487ad12

    # Binary Relevance Scheme
    # You basically train a classifier for each tag with as prediction 0 or 1 for each given tag.

    # Classifier Chain Scheme
    # Same as binary but the predictions of the previous classifiers are an extra feature for the next one.

    # Hamming Loss
    # Instead of counting no of correctly classified data instance, Hamming Loss calculates loss generated in the bit
    # string of class labels during prediction. It does XOR operation between the original binary string of class
    # labels and predicted class labels for a data instance and calculates the average across the dataset.

    # Subset Accuracy
    # There are some situations where we may go for an absolute accuracy ratio where measuring the exact combination
    # of label predictions is important.

    model = MultiOutputClassifier(estimator=RandomForestRegressor(n_estimators=300))
    grid_search_cv = GridSearchCV(model,
                                  rf_hyperparameters,
                                  cv=KFold(2, shuffle=True),
                                  scoring='neg_root_mean_squared_error',
                                  n_jobs=-1,
                                  return_train_score=True)

    print("Starting grid search now.\n")
    grid_search_cv.fit(x_train, y_train)

    best_parameters = grid_search_cv.best_params_
    print(f"Best mean squared score:{grid_search_cv.best_score_} with params:{best_parameters}\n")


if __name__ == '__main__':
    print("Starting unsupervised learning script.\n")
    remove_last_generated_results()

    json_questions = load_cached_questions()

    questions = [{
        "body": question['body'],
        "tags": question['tags'],
        "title": question['title']
    } for question in json_questions]

    print(f"{len(questions)} questions loaded from cache.\n")

    questions = list(map(extract_and_clean_text, questions))

    perform_supervised_modeling(questions)
