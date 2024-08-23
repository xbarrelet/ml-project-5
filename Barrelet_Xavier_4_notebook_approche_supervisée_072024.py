import json
import os
import shutil
import time
import warnings

import gensim.parsing.preprocessing as gsp
import joblib
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
from transformers import *
from xgboost import XGBModel, XGBClassifier

warnings.filterwarnings("ignore", category=DeprecationWarning)

# MISC CONFIGURATION
plt.style.use("fivethirtyeight")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 15k crashes during hyperoptimization with doc2vec already. 20k crashes with no hyperoptimization and USE.
NUMBER_OF_QUESTIONS_USED_IN_TRAINING = 10000

# PATHS
CACHED_QUESTIONS_FILE = 'cached_questions.json'
RESULTS_PATH = 'supervised_results'
MODELS_PATH = 'models/supervised'

# NLTK PACKAGES
nltk.download('wordnet')

# NLTK OBJECTS
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
    shutil.rmtree(MODELS_PATH, ignore_errors=True)
    os.mkdir(RESULTS_PATH)
    os.makedirs(MODELS_PATH, exist_ok=True)


def extract_and_clean_text(question: dict):
    """Create a new 'text' field for each question containing the cleaned, tokenized and lemmatized title + body."""
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


def bert_inp_fct(sentences, bert_tokenizer, max_length):
    """Returns BERT variables for its prediction."""
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


def transform_text_using_BERT(model, model_type, sentences, max_length, b_size):
    """Transform the text of the question's body and title into BERT embeddings."""
    # We don't want to use the cleaned text field with BERT, only title + " " + body
    sentences = [f"{sentence[1]} {sentence[0]}" for sentence in sentences.iterrows()]

    batch_size = b_size
    batch_size_pred = b_size
    bert_tokenizer = AutoTokenizer.from_pretrained(model_type)
    time1 = time.time()

    for step in range(len(sentences) // batch_size):
        idx = step * batch_size
        input_ids, token_type_ids, attention_mask, bert_inp_tot = bert_inp_fct(sentences[idx:idx + batch_size],
                                                                               bert_tokenizer, max_length)
        outputs = model.predict([input_ids, attention_mask, token_type_ids], batch_size=batch_size_pred)
        last_hidden_states = outputs.last_hidden_state

        if step == 0:
            last_hidden_states_tot = last_hidden_states
        else:
            last_hidden_states_tot = np.concatenate((last_hidden_states_tot, last_hidden_states))

    features_bert = np.array(last_hidden_states_tot).mean(axis=1)

    time2 = np.round(time.time() - time1, 0)
    print(f"BERT processing time:{time2}s\n")

    return features_bert


def transform_text_using_USE(sentences, b_size):
    """Transform the text of the question's body and title into USE embeddings."""
    # We don't want to use the cleaned text field with USE, only title + " " + body
    sentences = [f"{sentence[1]} {sentence[0]}" for sentence in sentences.iterrows()]

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
    """Transform the question text/body and title into words embeddings."""
    if text_transformation_method == "Doc2VEC":
        return transform_text_using_Doc2VEC(questions_without_tags["text"])

    elif text_transformation_method == "BERT":
        max_length = 64
        batch_size = 10
        model_type = 'bert-base-uncased'
        model = TFAutoModel.from_pretrained(model_type)

        return transform_text_using_BERT(model, model_type, questions_without_tags, max_length, batch_size)

    elif text_transformation_method == "USE":
        batch_size = 10
        return transform_text_using_USE(questions_without_tags, batch_size)


def transform_text_using_Doc2VEC(questions_without_tags):
    """Transform the text of the question's body and title into Doc2VEC embeddings."""
    time1 = time.time()
    tagged_text = [TaggedDocument(words=text, tags=[str(index)])
                   for index, text in enumerate(questions_without_tags)]

    # dm=0 for DBOW, dm=1 for PV-DM
    model = Doc2Vec(vector_size=30, min_count=2, epochs=80, dm=0)
    model.build_vocab(tagged_text)
    model.train(tagged_text, total_examples=model.corpus_count, epochs=model.epochs)

    embedded_text = [model.infer_vector(text.split(" ")) for text in questions_without_tags]

    time2 = np.round(time.time() - time1, 0)
    print(f"Doc2VEC processing time:{time2}s\n")
    return embedded_text


def create_results_plot(results):
    """Generate the plot showing the performance with each words embedding method."""
    performance_plot = results.plot(kind="bar", x="words_embedding_method", figsize=(15, 8), rot=0,
                                    title="Models Performance Sorted by Jaccard Score")
    performance_plot.legend([f"Hamming Loss", f"Jaccard Score"])
    performance_plot.title.set_size(20)
    performance_plot.set(xlabel=None)

    performance_plot.get_figure().savefig(f"{RESULTS_PATH}/performance_plot.png", bbox_inches='tight')
    plt.close()


def perform_supervised_modeling(questions):
    """Find the best model using a GridSearchCV hyperoptimization for each words embedding method."""
    questions_df = DataFrame(questions).head(NUMBER_OF_QUESTIONS_USED_IN_TRAINING)

    tags = MultiLabelBinarizer().fit_transform(questions_df['tags'])
    questions_df['tags'].to_json(f"{MODELS_PATH}/tags.json")

    questions_without_tags = questions_df.drop(columns=['tags'], axis=1)

    results_df = DataFrame(columns=["words_embedding_method", "hamming_loss", "jaccard_score"])
    models = {}
    for words_embedding_method in [
        # "Doc2VEC",
        # "BERT",
        "USE"
    ]:
        print(f"Starting supervised learning of {NUMBER_OF_QUESTIONS_USED_IN_TRAINING} questions with words embedding method:{words_embedding_method}.\n")
        transformed_text = transform_text(questions_without_tags, words_embedding_method)

        x_train, x_test, y_train, y_test = train_test_split(transformed_text, tags, test_size=0.2, random_state=42)
        print(f"training set size:{len(x_train)}, test set size:{len(x_test)}\n")

        # Best hyperparameters for 10k questions
        # default_model = RandomForestClassifier(n_estimators=100, max_depth=9, max_features=5)
        # default_hyperparameters = {'estimator__max_depth': range(2, 10), 'estimator__max_features': range(2, 6)}

        default_model = XGBClassifier(n_estimators=100, max_depth=9)
        default_hyperparameters = {'estimator__max_depth': range(2, 10)}

        grid_search_cv = GridSearchCV(MultiOutputClassifier(estimator=default_model), default_hyperparameters,
                                      cv=2,
                                      scoring=make_scorer(metrics.jaccard_score, average='samples'),
                                      n_jobs=-1,
                                      verbose=3
                                      )
        grid_search_cv.fit(x_train, y_train)
        # default_model.fit(x_train, y_train)

        best_parameters = grid_search_cv.best_params_
        print(f"\nBest mean squared score:{grid_search_cv.best_score_} with params:{best_parameters}")

        # best_model = default_model
        best_model = grid_search_cv.best_estimator_
        models[words_embedding_method] = best_model

        predictions_test_y = best_model.predict(x_test)

        hamming_loss = metrics.hamming_loss(y_true=y_test, y_pred=predictions_test_y)
        jaccard_score = metrics.jaccard_score(y_true=y_test, y_pred=predictions_test_y, average='samples')
        print(f"Hamming loss:{hamming_loss}, jaccard_score:{jaccard_score}\n")

        joblib.dump(best_model, f"{MODELS_PATH}/best_supervised_model.model")

        results_df.loc[len(results_df)] = [words_embedding_method, hamming_loss, jaccard_score]

        # send_results_to_mlflow(default_hyperparameters, best_model, hamming_loss, jaccard_score,
        #                        words_embedding_method, x_train)

    results_df.sort_values(f"jaccard_score", ascending=False, inplace=True)
    create_results_plot(results_df)

    save_best_model(models, results_df)


def save_best_model(models, results_df):
    """Save the best model based on the hamming loss."""
    best_words_embedding_method = results_df.head(1)['words_embedding_method'].values[0]
    joblib.dump(models[best_words_embedding_method], f"{MODELS_PATH}/best_supervised_model.model")


def send_results_to_mlflow(default_hyperparameters, best_model, hamming_loss, jaccard_score, words_embedding_method,
                           x_train):
    """Send data to the MLFlow server."""
    with mlflow.start_run():
        mlflow.log_params(default_hyperparameters)

        mlflow.log_metric("hamming_loss", hamming_loss)
        mlflow.log_metric("jaccard_score", jaccard_score)

        mlflow.set_tag("Words embedding method", words_embedding_method)

        signature = infer_signature(x_train, best_model.predict(x_train))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="supervised-models",
            signature=signature,
            input_example=x_train,
            registered_model_name="RandomForestClassifier",
        )


if __name__ == '__main__':
    print("Starting supervised learning script. Please make sure you have a local MLFlow server running.\n")
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
