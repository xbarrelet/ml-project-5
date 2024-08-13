import json
import os
import shutil
import string
import warnings
from pprint import pprint

import gensim
import gensim.parsing.preprocessing as gsp
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim import corpora
from gensim.models import CoherenceModel
from nltk import WordNetLemmatizer, PorterStemmer
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

# MISC CONFIGURATION
plt.style.use("fivethirtyeight")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# COHERENCE METRIC USED FOR HYPERPARAMETER OPTIMIZATION.
COHERENCE_METRIC = "u_mass"

# PATHS
CACHED_QUESTIONS_FILE = 'cached_questions.json'
RESULTS_PATH = 'unsupervised_results'

# NLTK PACKAGES
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('words')
# nltk.download('wordnet')

# NLTK OBJECTS
stopwords = nltk.corpus.stopwords.words('english')
words = set(nltk.corpus.words.words())
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def load_cached_questions():
    """Load questions from the cache file."""
    with open(CACHED_QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


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
    words_without_whitespaces = (gsp.strip_multiple_whitespaces(word) for word in words_without_short_words)

    # words_stemmed = (stemmer.stem(w) for w in words_without_short_words)
    words_lemmatized = (lemmatizer.lemmatize(w) for w in words_without_whitespaces)
    cleaned_text = ' '.join(w for w in words_lemmatized if w in words or not w.isalpha())
    question['text'] = cleaned_text

    # bigrams = nltk.bigrams(tokenized_text)
    # question['bigrams'] = [' '.join(bigram) for bigram in bigrams]

    # trigrams = nltk.trigrams(tokenized_text)
    # question['trigrams'] = [' '.join(trigram) for trigram in trigrams]

    return question


def compute_coherence_values_of_lda_model(corpus, id2word, texts, num_topics, alpha, eta):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           # passes=1,
                                           passes=10,
                                           alpha=alpha,
                                           eta=eta)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence=COHERENCE_METRIC)

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

    print("Visualizing the topics of the LDA model.\n")
    visualize_lda_topics(corpus, id2word, lda_model, best_hyperparameters['num_topics'])

    print("Saving the LDA model.\n")
    save_model(best_hyperparameters, lda_model)


def save_model(best_hyperparameters, lda_model):
    os.makedirs('models/unsupervised', exist_ok=True)
    lda_model.save(f"models/lda/lda_model_with_{best_hyperparameters['num_topics']}_topics.model")


def get_best_hyperparameters_of_lda_model(corpus, id2word, texts):
    """Returns the best hyperparameters for the LDA model based on the coherence metric."""
    topics_range = range(2, 12, 1)

    alphas = list(np.arange(0.01, 2, 0.3))
    alphas.append('symmetric')
    alphas.append('asymmetric')

    etas = list(np.arange(0.01, 1, 0.3))
    etas.append('symmetric')
    etas.append(None)

    model_results = []

    # OVERRIDES for test
    # alphas = ['symmetric']
    # etas = [None]

    pbar = tqdm(total=(len(etas) * len(alphas) * len(topics_range)))
    for num_topics in topics_range:
        for alpha in alphas:
            for eta in etas:
                cv = compute_coherence_values_of_lda_model(corpus, id2word, texts, num_topics, alpha, eta)

                model_results.append({"num_topics": num_topics, "alpha": alpha, "eta": eta, "cv": cv})
                pbar.update(1)

    pbar.close()

    print("Results of the hyperparameters search:")
    pprint(model_results)

    return max(model_results, key=lambda x: x['cv'])


def visualize_lda_topics(corpus, id2word, lda_model, num_topics):
    LDAvis_prepared = gensimvis.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(LDAvis_prepared, f'{RESULTS_PATH}/lda_results_with_{num_topics}_topics.html')


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
    print("Texts extracted and cleaned.\n")

    train_lda_model(questions)

    print("\nUnsupervised learning script finished.")