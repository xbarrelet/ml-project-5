import json
import os
import shutil
import string
import warnings
from collections import defaultdict
from datetime import datetime
from pprint import pprint
import seaborn as sns
import matplotlib.patches as mpatches

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk import WordNetLemmatizer
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from stackapi import StackAPI
from wordcloud import WordCloud

warnings.filterwarnings("ignore", category=DeprecationWarning)

# STACKAPI CONFIGURATION
SITE = StackAPI('stackoverflow')
# Max and default value is 100
# SITE.page_size = 100
# More than 25 pages requires an API key
SITE.max_pages = 25

# MISC CONFIGURATION
plt.style.use("fivethirtyeight")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
RESULTS_PATH = 'analysis_results'

# NLTK PACKAGES
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
                           fromdate=datetime(2020, 1, 1),
                           todate=datetime(2024, 7, 30),
                           min=50,
                           sort='votes',
                           filter='withbody',
                           # tagged='python'
                           )

    with open('cached_questions.json', 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)


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

    # Keeping only the common part of verbs for example
    words_lemmatized = (lemmatizer.lemmatize(w) for w in words_without_stopwords)
    cleaned_text = ' '.join(w for w in words_lemmatized if w in words or not w.isalpha())
    question['text'] = cleaned_text

    bigrams = nltk.bigrams(tokenized_text)
    question['bigrams'] = [' '.join(bigram) for bigram in bigrams]

    trigrams = nltk.trigrams(tokenized_text)
    question['trigrams'] = [' '.join(trigram) for trigram in trigrams]

    return question


def visualize_word_cloud(texts: list[str]):
    joined_texts = ','.join(texts)
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3,
                          contour_color='steelblue')
    cloud = wordcloud.generate(joined_texts)
    cloud.to_file(f"{RESULTS_PATH}/wordcloud.png")


def display_number_of_words_per_tag(questions):
    words_per_tag = defaultdict(list)

    for question in questions:
        for tag in question['tags']:
            words_per_tag[tag] += question['text'].split(" ")

    tags = []
    for tag in words_per_tag.keys():
        tags.append({
            "tag": tag,
            "words": len(words_per_tag[tag]),
            "unique_words": len(list(set(words_per_tag[tag])))
        })

    df: DataFrame = pd.DataFrame(tags)
    df.sort_values(by='words', ascending=False, inplace=True)
    top_50_df = df.head(50)

    generate_plot_with_words_per_tag(top_50_df)


def generate_plot_with_words_per_tag(top_50_df):
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(14, 14)

    sns.barplot(x="words", y="tag", data=top_50_df, color='#fce0cc', ax=axs)
    sns.barplot(x="unique_words", y="tag", data=top_50_df, color='#f56900', ax=axs)
    plt.title('Number of unique words of the top 50 tags')

    total_bar = mpatches.Patch(color='#fce0cc', label='Total')
    unique_words_bar = mpatches.Patch(color='#f56900', label='Unique words')
    fig.legend(handles=[total_bar, unique_words_bar])

    fig.savefig(f"{RESULTS_PATH}/words_per_tag.png", bbox_inches='tight')


def display_length_of_body_and_title(questions):
    df: DataFrame = pd.DataFrame(questions)

    df['body_length'] = df['body'].apply(lambda x: len(x))
    df['title_length'] = df['title'].apply(lambda x: len(x))

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(14, 7)

    sns.histplot(df['body_length'], kde=True, ax=axs[0], color='#f56900')
    sns.histplot(df['title_length'], kde=True, ax=axs[1], color='#f56900')

    axs[0].set_title('Distribution of the body length')
    axs[1].set_title('Distribution of the title length')

    fig.savefig(f"{RESULTS_PATH}/length_of_body_and_title.png", bbox_inches='tight')
    plt.close()

    plot = sns.boxplot(df, x="body_length")
    plot.get_figure().savefig(f"{RESULTS_PATH}/length_of_body_boxplot.png", bbox_inches='tight')


def perform_tf_idf_analysis(questions):
    all_texts = [question['text'] for question in questions]
    tfidf = TfidfVectorizer()
    values = tfidf.fit_transform(all_texts)
    print(values)

    # Should I do that manually instead for a better visibility?


if __name__ == '__main__':
    print("Starting project 5!\n")
    remove_last_generated_results()

    # cache_questions()

    json_questions = load_cached_questions()

    questions = [{
        "body": question['body'],
        "tags": question['tags'],
        "title": question['title']
    } for question in json_questions]

    """Un notebook d’exploration et de pré-traitement des questions comprenant 
    une analyse univariée et multivariée, 
    un nettoyage des questions, 
    un feature engineering de type bag of words avec réduction de dimension (du vocabulaire et des tags) """

    print(f"{len(questions)} questions loaded from cache.\n")

    df: DataFrame = DataFrame(questions)

    # Lot of false positives with the technical language, you could do it in the final dataset if enough rows remain
    # non_english_questions = [question for question in questions
    #                          if detect(question['title']) != 'en' and detect(question['body']) != 'en' ]
    # print(f"{len(non_english_questions)} non-english questions were found but most are false positives.\n")
    # pprint(non_english_questions)

    questions = list(map(extract_and_clean_text, questions))

    # display_length_of_body_and_title(questions)

    # display_number_of_words_per_tag(questions)

    # visualize_word_cloud([question['text'] for question in questions])
    
    perform_tf_idf_analysis(questions)

