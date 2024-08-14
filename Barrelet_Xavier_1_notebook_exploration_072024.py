import json
import os
import shutil
import warnings
from collections import defaultdict
from datetime import datetime
from os.path import exists

import gensim.parsing.preprocessing as gsp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from nltk import WordNetLemmatizer, PorterStemmer
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE
from stackapi import StackAPI
from wordcloud import WordCloud

warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# STACKAPI CONFIGURATION
if os.getenv("SITE_API_KEY") is None:
    SITE = StackAPI('stackoverflow')
    # 25 = limit with default of 100 results per page and no api key
    SITE.max_pages = 25
else:
    SITE = StackAPI('stackoverflow', key=os.getenv("SITE_API_KEY"))
    # To get 50k results
    SITE.max_pages = 500

# MISC CONFIGURATION
plt.style.use("fivethirtyeight")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# PATHS
CACHED_QUESTIONS_FILE = 'cached_questions.json'
RESULTS_PATH = 'analysis_results'

# NLTK PACKAGES
nltk.download('wordnet')

# NLTK OBJECTS
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def load_cached_questions():
    with open(CACHED_QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def cache_questions():
    # https://stackapi.readthedocs.io/en/latest/user/complex.html
    questions = SITE.fetch('questions',
                           fromdate=datetime(2010, 1, 1),
                           todate=datetime(2024, 8, 11),
                           min=50,
                           sort='votes',
                           filter='withbody',
                           # tagged='python'
                           )

    extracted_questions = questions['items']
    trimmed_questions = [{
        "body": question['body'],
        "creation_date": question['creation_date'],
        "score": question['score'],
        "tags": question['tags'],
        "title": question['title']
    } for question in extracted_questions]

    with open(CACHED_QUESTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(trimmed_questions, f, ensure_ascii=False, indent=4)


def remove_last_generated_results():
    """Removes the content of the saved plots."""
    shutil.rmtree(RESULTS_PATH, ignore_errors=True)
    os.mkdir(RESULTS_PATH)


def extract_and_clean_text(question: dict):
    title = question['title']
    body = question['body']
    text = f"{title} {body}"

    for filter in [gsp.strip_tags,
                   gsp.strip_punctuation,
                   gsp.strip_multiple_whitespaces,
                   gsp.strip_numeric,
                   gsp.remove_stopwords,
                   gsp.strip_short,
                   gsp.lower_to_unicode]:
        text = filter(text)

    tokenized_text = nltk.tokenize.word_tokenize(text)

    # words_stemmed = (stemmer.stem(w) for w in words_without_short_words)
    words_lemmatized = [lemmatizer.lemmatize(w) for w in tokenized_text]
    question['text'] = " ".join(words_lemmatized)

    # bigrams = nltk.bigrams(tokenized_text)
    # question['bigrams'] = [' '.join(bigram) for bigram in bigrams]

    # trigrams = nltk.trigrams(tokenized_text)
    # question['trigrams'] = [' '.join(trigram) for trigram in trigrams]

    return question


def visualize_word_clouds(questions):
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3,
                          contour_color='steelblue')

    generate_words_wordcloud(questions, wordcloud)
    generate_tags_wordcloud(questions, wordcloud)


def generate_tags_wordcloud(questions, wordcloud):
    unique_tags = set([tag for question in questions for tag in question['tags']])
    joined_tags = ','.join(unique_tags)
    print(f"{len(unique_tags)} unique tags were found in the dataset.\n")

    cloud = wordcloud.generate(joined_tags)

    cloud.to_file(f"{RESULTS_PATH}/tags_wordcloud.png")


def generate_words_wordcloud(questions, wordcloud):
    texts = [question['text'] for question in questions]
    joined_texts = ','.join(texts)

    cloud = wordcloud.generate(joined_texts)

    cloud.to_file(f"{RESULTS_PATH}/words_wordcloud.png")


def display_most_used_tags(questions):
    top_50_df = get_most_used_tags(questions, 50)

    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(14, 14)

    sns.barplot(x="count", y="tag", data=top_50_df, color='#f56900', ax=axs)
    plt.title('Most used tags')

    fig.savefig(f"{RESULTS_PATH}/most_used_tags.png", bbox_inches='tight')
    plt.close()


def get_most_used_tags(questions, count):
    tags = defaultdict(int)

    for question in questions:
        for tag in question['tags']:
            tags[tag] += 1

    df: DataFrame = pd.DataFrame(list(tags.items()), columns=['tag', 'count'])
    df.sort_values(by='count', ascending=False, inplace=True)

    return df.head(count)


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
    plt.close()


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

    boxplot = sns.boxplot(df, x="body_length")
    boxplot.get_figure().savefig(f"{RESULTS_PATH}/length_of_body_boxplot.png", bbox_inches='tight')
    plt.close()

    boxplot_without_outliers = sns.boxplot(df, x="body_length", showfliers=False)
    boxplot_without_outliers.get_figure().savefig(f"{RESULTS_PATH}/length_of_body_boxplot_without_outliers.png",
                                                  bbox_inches='tight')
    plt.close()


def visualize_dimensionality_reductions(questions):
    five_most_used_tags = get_most_used_tags(questions, 5)['tag']
    all_texts = [question['text'] for question in questions]

    tfidf_vectorizer_model = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=1)
    tfidf_vectorizer_model.fit(all_texts)

    count_vectorizer_model = CountVectorizer(stop_words='english', max_df=0.95, min_df=1)
    count_vectorizer_model.fit(all_texts)

    texts_df: DataFrame = DataFrame(columns=['tag', 'tsne-x-tfidf', 'tsne-y-tfidf', 'tsne-x-cv', 'tsne-y-cv'])

    for tag in five_most_used_tags:
        texts_for_tag = [question['text'] for question in questions if tag in question['tags']]

        tsne_results_tfidf = get_tsne_results_with_model(texts_for_tag, tfidf_vectorizer_model)
        tsne_results_cv = get_tsne_results_with_model(texts_for_tag, count_vectorizer_model)

        for index in range(len(tsne_results_tfidf)):
            texts_df.loc[len(texts_df)] = [tag,
                                           tsne_results_tfidf[index][0], tsne_results_tfidf[index][1],
                                           tsne_results_cv[index][0], tsne_results_cv[index][1]]

    create_tsne_visualization_for_model(texts_df, "tfidf")
    create_tsne_visualization_for_model(texts_df, "cv")


def create_tsne_visualization_for_model(texts_df: DataFrame, model_type: str) -> None:
    plt.figure(figsize=(12, 10))

    ctf_plot = sns.scatterplot(texts_df,
                               x=f"tsne-x-{model_type}", y=f"tsne-y-{model_type}",
                               hue="tag", palette="bright")

    model_name = "TfidfVectorizer" if model_type == "tfidf" else "CountVectorizer"
    ctf_plot.set_title(f'Scatter plot of the texts linked to the five most used tags with {model_name}')

    ctf_plot.set_xlabel(f'F1')
    ctf_plot.set_ylabel(f'F2')
    ctf_plot.grid(True)

    ctf_plot.get_figure().savefig(f"{RESULTS_PATH}/tsne_{model_type}.png", bbox_inches='tight')
    plt.close()


def get_tsne_results_with_model(texts_for_tag, tfidf_vectorizer_model):
    ctf_transformed_text = tfidf_vectorizer_model.transform(texts_for_tag)

    tsne_ctf = TSNE(n_components=2, perplexity=30, max_iter=2000, init='random', learning_rate=200, random_state=42)
    return tsne_ctf.fit_transform(ctf_transformed_text)


if __name__ == '__main__':
    print("Starting analysis script.\n")
    remove_last_generated_results()

    if not exists(CACHED_QUESTIONS_FILE):
        print(f"Cached questions are missing, downloading them in {CACHED_QUESTIONS_FILE}.\n")
        cache_questions()

    json_questions = load_cached_questions()

    questions = [{
        "body": question['body'],
        "tags": question['tags'],
        "title": question['title']
    } for question in json_questions]
    print(f"{len(questions)} questions loaded from cache.\n")

    # Too many false positives due to included code or technical words.
    # non_english_questions = [question for question in questions if langdetect.detect(question['body']) != 'en']

    cleaned_questions = list(map(extract_and_clean_text, questions))
    print(f"Texts extracted and cleaned.\n")

    display_length_of_body_and_title(cleaned_questions)
    print("Length of body and title displayed.\n")

    display_most_used_tags(cleaned_questions)
    print("Most used tags displayed.\n")

    display_number_of_words_per_tag(cleaned_questions)
    print("Number of words per tag displayed.\n")

    visualize_word_clouds(cleaned_questions)
    print("Word clouds displayed.\n")

    visualize_dimensionality_reductions(cleaned_questions)
    print("Dimensionality reductions displayed.\n")

    print("Analysis script finished.\n")
