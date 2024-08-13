import json
import os
import shutil
import string
import warnings
from datetime import datetime

import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim import corpora
from gensim.models import CoherenceModel
from nltk import WordNetLemmatizer
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from stackapi import StackAPI
from tqdm import tqdm
from wordcloud import WordCloud

warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    with open('cached_questions_BAK.json', 'r', encoding='utf-8') as f:
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

    with open('cached_questions_BAK.json', 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)


def remove_last_generated_results():
    """Removes the content of the saved plots."""
    shutil.rmtree('results', ignore_errors=True)
    os.mkdir('results')


def extract_and_clean_text(question: dict):
    title = question['title']
    body = question['body']
    text = f"{title} {body}"

    text_without_punctuation = "".join([i.lower() for i in text if i not in string.punctuation])
    text_without_number = ''.join(i for i in text_without_punctuation if not i.isdigit())

    tokenized_text = nltk.tokenize.word_tokenize(text_without_number)
    words_without_stopwords = [i for i in tokenized_text if i not in stopwords]

    words_lemmatized = (lemmatizer.lemmatize(w) for w in words_without_stopwords)
    cleaned_text = ' '.join(w.lower() for w in words_lemmatized if w.lower() in words or not w.isalpha())

    question['text'] = cleaned_text
    return question


def visualize_word_cloud(texts: list[str]):
    joined_texts = ','.join(texts)
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3,
                          contour_color='steelblue')
    cloud = wordcloud.generate(joined_texts)
    cloud.to_file("results/wordcloud.png")


def compute_coherence_values_of_lda_model(corpus, id2word, texts, num_topics, alpha, eta):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           # passes=1,
                                           passes=10,
                                           alpha=alpha,
                                           eta=eta)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')

    return coherence_model_lda.get_coherence()


def train_lda_model(questions):
    print("Starting the search of the best hyperparameters of the LDA model.\n")
    texts = [question['text'].split(" ") for question in questions]

    id2word = corpora.Dictionary(texts)
    corpus = [id2word.doc2bow(text) for text in texts]

    best_hyperparameters: dict = get_best_hyperparameters_of_lda_model(corpus, id2word, texts)
    print(f"Best hyperparameters found:{best_hyperparameters}.\n")

    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=best_hyperparameters['num_topics'],
                                           passes=10,
                                           alpha=best_hyperparameters['alpha'],
                                           eta=best_hyperparameters['eta'])

    visualize_lda_topics(corpus, id2word, lda_model, best_hyperparameters['num_topics'])

    """    
    C_v measure is based on a sliding window, one-set segmentation of the top words and an indirect confirmation measure that uses normalized pointwise mutual information (NPMI) and the cosine similarity
    C_p is based on a sliding window, one-preceding segmentation of the top words and the confirmation measure of Fitelson’s coherence
    C_uci measure is based on a sliding window and the pointwise mutual information (PMI) of all word pairs of the given top words
    C_umass is based on document cooccurrence counts, a one-preceding segmentation and a logarithmic conditional probability as confirmation measure
    C_npmi is an enhanced version of the C_uci coherence using the normalized pointwise mutual information (NPMI)
    C_a is baseed on a context window, a pairwise comparison of the top words and an indirect confirmation measure that uses normalized pointwise mutual information (NPMI) and the cosine similarity
    """

    # TODO: Evaluate it: https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0


def get_best_hyperparameters_of_lda_model(corpus, id2word, texts):
    topics_range = range(2, 12, 1)

    alphas = list(np.arange(0.01, 1, 0.3))
    alphas.append('symmetric')
    alphas.append('asymmetric')

    etas = list(np.arange(0.01, 1, 0.3))
    etas.append('symmetric')
    etas.append(None)

    model_results = []

    # OVERRIDES for test
    alphas = ['symmetric']
    etas = [None]

    pbar = tqdm(total=(len(etas) * len(alphas) * len(topics_range)))
    for num_topics in topics_range:
        for alpha in alphas:
            for eta in etas:
                cv = compute_coherence_values_of_lda_model(corpus, id2word, texts, num_topics, alpha, eta)

                model_results.append({"num_topics": num_topics, "alpha": alpha, "eta": eta, "cv": cv})
                pbar.update(1)

    pbar.close()
    return max(model_results, key=lambda x: x['cv'])


def visualize_lda_topics(corpus, id2word, lda_model, num_topics):
    LDAvis_prepared = gensimvis.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(LDAvis_prepared, f'results/lda_results_with_{num_topics}_topics.html')


def perform_supervised_modeling(questions):
    questions_df = DataFrame(questions)

    questions_without_tags = questions_df.drop(columns=['tags'], axis=1)
    tags = questions_df['tags']

    x_train, x_test, y_train, y_Test = train_test_split(questions_without_tags, tags, test_size=0.2,
                                                        random_state=42)

    pass


if __name__ == '__main__':
    print("Starting project 5!\n")
    remove_last_generated_results()

    # cache_questions()

    json_questions = load_cached_questions()

    questions = [{
        "body": question['body'],
        "creation_date": question['creation_date'],
        "tags": question['tags'],
        "title": question['title'],
        "score": question['score']
    } for question in json_questions]

    print(f"{len(questions)} questions loaded from cache.\n")

    df: DataFrame = DataFrame(questions)

    # Lot of false positives with the technical language, you could do it in the final dataset if enough rows remain
    # non_english_questions = [question for question in questions
    #                          if detect(question['title']) != 'en' and detect(question['body']) != 'en' ]
    # print(f"{len(non_english_questions)} non-english questions were found but most are false positives.\n")
    # pprint(non_english_questions)

    questions = list(map(extract_and_clean_text, questions))

    visualize_word_cloud([question['text'] for question in questions])

    # There are several existing algorithms you can use to perform the topic modeling. The most common of it are:
    # Latent Semantic Analysis (LSA/LSI), Probabilistic Latent Semantic Analysis (pLSA) and Latent Dirichlet Allocation (LDA)
    # LDA suffit pour ce projet cependant
    # train_lda_model(questions)

    perform_supervised_modeling(questions)

    # Supervized = classification avec plusieurs classes comme prediction, pas single vu que tu peux avoir plusieurs tags
    # Reprend le notebook pour extraire les features comme ils veulent, pas besoin de custom transformers ici
    # https://scikit-learn.org/stable/modules/multiclass.html pour wrapper autour dun randomforest.
    # les tags vont donner une liste de vecteurs qui sera ma colonne de prediction

    # une approche de type bag-of-words (Bag of word - Tf-idf)

    # Puis
    # 3 approches de Word/Sentence Embedding : Word2Vec (ou Doc2Vec, Glove…), BERT et USE.

    # Unsupervized

    # Feature extraction from text: https://medium.com/@eskandar.sahel/exploring-feature-extraction-techniques-for-natural-language-processing-46052ee6514
    # CountVectorizer, TF-IDF, word embeddings, bag of words, bag of n-grams, HashingVectorizer, Latent Dirichlet Allocation (LDA),
    # Non-negative Matrix Factorization (NMF), Principal Component Analysis (PCA), t-SNE, and Part-of-Speach (POS) tagging.

    # Also check the notebook in this folder for examples from Openclassrooms
