import json
import os
import pickle
import shutil
import string
from datetime import datetime
from pprint import pprint

import gensim
import pyLDAvis
from gensim import corpora
from nltk import WordNetLemmatizer
from pandas import DataFrame
from stackapi import StackAPI
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect
import nltk
from wordcloud import WordCloud
import pyLDAvis.gensim_models as gensimvis

SITE = StackAPI('stackoverflow')

# Max and default value is 100
# SITE.page_size = 100
# More than 25 pages requires an API key
SITE.max_pages = 25

plt.style.use("fivethirtyeight")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('words')
# nltk.download('wordnet')

stopwords = nltk.corpus.stopwords.words('english')
words = set(nltk.corpus.words.words())
lemmatizer = WordNetLemmatizer()


def load_cached_questions():
    with open('cached_questions.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        return json_data['items']


def cache_questions():
    # https://stackapi.readthedocs.io/en/latest/user/complex.html
    questions = SITE.fetch('questions',
                           fromdate=datetime(2010, 1, 1),
                           todate=datetime(2024, 7, 30),
                           min=50,
                           sort='votes',
                           tagged='python')

    with open('cached_questions.json', 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)


def remove_last_generated_results():
    """Removes the content of the saved plots."""
    shutil.rmtree('results', ignore_errors=True)
    os.mkdir('results')


def add_cleaned_title(question: dict):
    title = question['original_title']

    title_without_punctuation = "".join([i.lower() for i in title if i not in string.punctuation])
    title_without_number = ''.join(i for i in title_without_punctuation if not i.isdigit())

    tokenized_title = nltk.tokenize.word_tokenize(title_without_number)
    words_without_stopwords = [i for i in tokenized_title if i not in stopwords]

    words_lemmatized = (lemmatizer.lemmatize(w) for w in words_without_stopwords)
    cleaned_title = ' '.join(w.lower() for w in words_lemmatized if w.lower() in words or not w.isalpha())

    question['title'] = cleaned_title
    return question


def visualize_word_cloud(titles: list[str]):
    joined_titles = ','.join(titles)
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3,
                          contour_color='steelblue')
    cloud = wordcloud.generate(joined_titles)
    cloud.to_file("results/wordcloud.png")


def train_lda_model(questions):
    titles = [question['title'].split(" ") for question in questions]

    id2word = corpora.Dictionary(titles)
    corpus = [id2word.doc2bow(title) for title in titles]

    num_topics = 10
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)
    # pprint(lda_model.show_topics())

    LDAvis_prepared = gensimvis.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_' + str(num_topics) + '.html')

    # TODO: Evaluate it: https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0


if __name__ == '__main__':
    print("Starting project 5!\n")
    remove_last_generated_results()

    # cache_questions()

    json_questions = load_cached_questions()

    questions = [{
        # No body for now, use the title only
        "creation_date": question['creation_date'],
        "tags": question['tags'],
        "original_title": question['title'],
        "score": question['score']
    } for question in json_questions]

    print(f"{len(questions)} questions loaded from cache.\n")

    df: DataFrame = DataFrame(questions)

    # Lot of false positives with the technical language, you could do it in the final dataset if enough rows remain
    non_english_questions = [question['original_title'] for question in questions
                             if detect(question['original_title']) != 'en']
    print(f"{len(non_english_questions)} non-english questions were found but most are false positives.\n")
    # pprint(non_english_questions)

    questions = list(map(add_cleaned_title, questions))

    visualize_word_cloud([question['title'] for question in questions])

    train_lda_model(questions)

    # Unsupervized
    # There are several existing algorithms you can use to perform the topic modeling. The most common of it are:
    # Latent Semantic Analysis (LSA/LSI), Probabilistic Latent Semantic Analysis (pLSA), and Latent Dirichlet Allocation (LDA)

    # Feature extraction from text: https://medium.com/@eskandar.sahel/exploring-feature-extraction-techniques-for-natural-language-processing-46052ee6514
    # CountVectorizer, TF-IDF, word embeddings, bag of words, bag of n-grams, HashingVectorizer, Latent Dirichlet Allocation (LDA),
    # Non-negative Matrix Factorization (NMF), Principal Component Analysis (PCA), t-SNE, and Part-of-Speach (POS) tagging.

    # Also check the notebook in this folder for examples from Openclassrooms
