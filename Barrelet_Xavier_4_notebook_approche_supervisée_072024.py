import json
import os
import shutil
import string
import time
import warnings
from pprint import pprint

import gensim.parsing.preprocessing as gsp
import matplotlib.pyplot as plt
import mlflow
import nltk
import numpy as np
import pandas as pd
import tensorflow_hub as hub
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from mlflow.models import infer_signature
from nltk import WordNetLemmatizer, PorterStemmer
from pandas import DataFrame
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain
from transformers import *
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=DeprecationWarning)

# MISC CONFIGURATION
plt.style.use("fivethirtyeight")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# PATHS
CACHED_QUESTIONS_FILE = 'cached_questions_2500.json'
RESULTS_PATH = 'supervised_results'

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

# To avoid having multiprocessing issues between BERT and the GridsearchCV
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# MLFlow
mlflow.set_tracking_uri(uri="http://localhost:8080")
mlflow.set_experiment("Supervised Learning Experiment")


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


def bert_inp_fct(sentences, bert_tokenizer, max_length):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    bert_inp_tot = []

    for sent in sentences:
        bert_inp = bert_tokenizer.encode_plus(sent,
                                              add_special_tokens=True,
                                              max_length=max_length,
                                              padding='max_length',
                                              return_attention_mask=True,
                                              return_token_type_ids=True,
                                              truncation=True,
                                              return_tensors="tf")

        input_ids.append(bert_inp['input_ids'][0])
        token_type_ids.append(bert_inp['token_type_ids'][0])
        attention_mask.append(bert_inp['attention_mask'][0])
        bert_inp_tot.append((bert_inp['input_ids'][0],
                             bert_inp['token_type_ids'][0],
                             bert_inp['attention_mask'][0]))

    input_ids = np.asarray(input_ids)
    token_type_ids = np.asarray(token_type_ids)
    attention_mask = np.array(attention_mask)

    return input_ids, token_type_ids, attention_mask, bert_inp_tot


def feature_BERT_fct(model, model_type, sentences, max_length, b_size, mode='HF'):
    batch_size = b_size
    batch_size_pred = b_size
    bert_tokenizer = AutoTokenizer.from_pretrained(model_type)
    time1 = time.time()

    for step in range(len(sentences) // batch_size):
        idx = step * batch_size
        input_ids, token_type_ids, attention_mask, bert_inp_tot = bert_inp_fct(sentences[idx:idx + batch_size],
                                                                               bert_tokenizer, max_length)

        if mode == 'HF':  # Bert HuggingFace
            outputs = model.predict([input_ids, attention_mask, token_type_ids], batch_size=batch_size_pred)
            last_hidden_states = outputs.last_hidden_state

        if mode == 'TFhub':  # Bert Tensorflow Hub
            text_preprocessed = {"input_word_ids": input_ids,
                                 "input_mask": attention_mask,
                                 "input_type_ids": token_type_ids}
            outputs = model(text_preprocessed)
            last_hidden_states = outputs['sequence_output']

        if step == 0:
            last_hidden_states_tot = last_hidden_states
        else:
            last_hidden_states_tot = np.concatenate((last_hidden_states_tot, last_hidden_states))

    features_bert = np.array(last_hidden_states_tot).mean(axis=1)

    time2 = np.round(time.time() - time1, 0)
    print(f"BERT processing time:{time2}s\n")

    return features_bert


def feature_USE_fct(sentences, b_size):
    batch_size = b_size
    time1 = time.time()
    us_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    for step in range(len(sentences) // batch_size):
        idx = step * batch_size
        feat = us_encoder(sentences[idx:idx + batch_size])

        if step == 0:
            features = feat
        else:
            features = np.concatenate((features, feat))

    time2 = np.round(time.time() - time1, 0)
    print(f"USE processing time:{time2}s\n")
    return features


def transform_text(questions_without_tags, text_transformation_method):
    # TODO: You could add a Doc2Vec avec dm=1 et Word2Vec for comparison

    if text_transformation_method == "Doc2VEC":
        return feature_Doc2VEC_fct(questions_without_tags)

    elif text_transformation_method == "BERT":
        max_length = 64
        batch_size = 10
        model_type = 'bert-base-uncased'
        model = TFAutoModel.from_pretrained(model_type)

        return feature_BERT_fct(model, model_type, questions_without_tags, max_length, batch_size)

    elif text_transformation_method == "USE":
        batch_size = 10
        return feature_USE_fct(questions_without_tags, batch_size)


def feature_Doc2VEC_fct(questions_without_tags):
    time1 = time.time()
    tagged_text = [TaggedDocument(words=text, tags=[str(index)])
                   for index, text in enumerate(questions_without_tags)]

    # dm=0 for DBOW, dm=1 for PV-DM
    model = Doc2Vec(vector_size=30, min_count=2, epochs=80, dm=0)
    model.build_vocab(tagged_text)
    model.train(tagged_text, total_examples=model.corpus_count, epochs=model.epochs)

    # model.save("d2v.model")
    embedded_text = [model.infer_vector(text.split(" ")) for text in questions_without_tags]

    time2 = np.round(time.time() - time1, 0)
    print(f"Doc2VEC processing time:{time2}s\n")
    return embedded_text


def perform_supervised_modeling(questions):
    questions_df = DataFrame(questions)[['text', 'tags']]

    tags = MultiLabelBinarizer().fit_transform(questions_df['tags'])
    questions_without_tags = questions_df.drop(columns=['tags'], axis=1)

    results = []
    for words_embedding_method in [
        "Doc2VEC",
        "BERT",
        "USE"
    ]:
        print(f"Starting supervised learning with words embedding method:{words_embedding_method}.\n")
        transformed_text = transform_text(questions_without_tags['text'], words_embedding_method)

        x_train, x_test, y_train, y_test = train_test_split(transformed_text, tags, test_size=0.2, random_state=42)
        print(f"training set size:{len(x_train)}, test set size:{len(x_test)}\n")

        # http://scikit.ml/modelselection.html

        result = {}
        default_model = XGBClassifier(n_estimators=100)
        default_hyperparameters = {'estimator__max_depth': range(2, 12), 'estimator__n_estimators': range(50, 151, 50)}

        grid_search_cv = GridSearchCV(MultiOutputClassifier(estimator=default_model), default_hyperparameters,
                                      cv=2,
                                      scoring=make_scorer(metrics.hamming_loss, greater_is_better=False),
                                      n_jobs=-1,
                                      # return_train_score=True,
                                      verbose=3
                                      )
        grid_search_cv.fit(x_train, y_train)

        best_parameters = grid_search_cv.best_params_
        print(f"\nBest mean squared score:{grid_search_cv.best_score_} with params:{best_parameters}")

        predictions_test_y = grid_search_cv.best_estimator_.predict(x_test)

        hamming_loss = metrics.hamming_loss(y_true=y_test, y_pred=predictions_test_y)
        jaccard_score = metrics.jaccard_score(y_true=y_test, y_pred=predictions_test_y, average='samples')
        print(f"Hamming loss:{hamming_loss}, jaccard_score:{jaccard_score}\n")

        result[words_embedding_method] = {
            "hamming_loss": hamming_loss,
            "jaccard_score": jaccard_score
        }

        with mlflow.start_run():
            mlflow.log_params(default_hyperparameters)

            mlflow.log_metric("hamming_loss", hamming_loss)
            mlflow.log_metric("jaccard_score", jaccard_score)

            mlflow.set_tag("Words embedding method", words_embedding_method)

            signature = infer_signature(x_train, grid_search_cv.best_estimator_.predict(x_train))

            mlflow.sklearn.log_model(
                sk_model=grid_search_cv.best_estimator_,
                artifact_path="supervised-models",
                signature=signature,
                input_example=x_train,
                registered_model_name="XGBoost",
            )

    print("Results:\n")
    pprint(results)


if __name__ == '__main__':
    print("Starting supervised learning script.\n")
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

    perform_supervised_modeling(questions)

    print("\nSupervised learning now finished.\n")

    # https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html
