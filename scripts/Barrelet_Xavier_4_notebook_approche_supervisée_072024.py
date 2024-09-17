import json
import multiprocessing
import os
import shutil
import time
import warnings

import cupy as cp
import gensim
import gensim.parsing.preprocessing as gsp
import joblib
import matplotlib.pyplot as plt
import mlflow
import nltk
import numpy as np
import pandas as pd
import tensorflow_hub as hub
from gensim.models import Word2Vec
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from keras import Input, Model
from keras.src.layers import Embedding, GlobalAveragePooling1D
from keras.src.utils import pad_sequences
from mlflow.models import infer_signature
from nltk import WordNetLemmatizer, PorterStemmer
from pandas import DataFrame
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from tf_keras.src.preprocessing.text import Tokenizer
from transformers import *
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=DeprecationWarning)

# MISC CONFIGURATION
plt.style.use("fivethirtyeight")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

NUMBER_OF_QUESTIONS_USED_IN_TRAINING = 100

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
# mlflow.set_tracking_uri(uri="http://localhost:8080")
# mlflow.set_experiment("Supervised Learning Experiment")


def load_cached_questions():
    """Load questions from the cache file."""
    with open(CACHED_QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def remove_last_generated_results():
    """Removes the content of the saved plots."""
    shutil.rmtree(RESULTS_PATH, ignore_errors=True)
    os.mkdir(RESULTS_PATH)
    shutil.rmtree(MODELS_PATH, ignore_errors=True)
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

    cleaned_text = text.replace("quot", "")
    tokenized_text = nltk.tokenize.word_tokenize(cleaned_text)

    # words_stemmed = (stemmer.stem(w) for w in words_without_short_words)
    words_lemmatized = [lemmatizer.lemmatize(w) for w in tokenized_text]
    question['text'] = " ".join(words_lemmatized)

    # bigrams = nltk.bigrams(tokenized_text)
    # question['bigrams'] = [' '.join(bigram) for bigram in bigrams]

    # trigrams = nltk.trigrams(tokenized_text)
    # question['trigrams'] = [' '.join(trigram) for trigram in trigrams]

    return question


def transform_text_using_BagOfWords(questions_without_tags, is_count_vectorizer=False):
    """Transform the text of the question's body and title into Word of bags embeddings."""
    time1 = time.time()

    model = CountVectorizer(stop_words='english', max_features=400) if is_count_vectorizer \
        else TfidfVectorizer(stop_words='english', max_features=400)

    embedded_text = model.fit_transform(questions_without_tags)

    joblib.dump(model, f"{MODELS_PATH}/embedder_model.model")

    time2 = np.round(time.time() - time1, 0)
    print(f"Word of bags processing time:{time2}s\n")
    return embedded_text.todense(), time2


def transform_text_using_Word2Vec(questions_without_tags):
    """Transform the text of the question's body and title into Doc2VEC embeddings."""
    time1 = time.time()

    sentences = questions_without_tags
    sentences = [gensim.utils.simple_preprocess(text) for text in sentences]

    vector_size = 300
    maxlen = max([len(sentence) for sentence in sentences])

    w2v_model = train_w2v_model(sentences, vector_size)
    model_vectors = w2v_model.wv
    w2v_words = model_vectors.index_to_key

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    x_sentences = pad_sequences(tokenizer.texts_to_sequences(sentences), maxlen=maxlen, padding='post')

    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1

    embedding_matrix = create_word2vec_embedding_matrix(model_vectors, vector_size, vocab_size, w2v_words, word_index)
    embedding_model = train_word2vec_embedding_model(embedding_matrix, maxlen, vector_size, vocab_size)

    embedded_text = embedding_model.predict(x_sentences)

    time2 = np.round(time.time() - time1, 0)
    print(f"Doc2VEC processing time:{time2}s\n")
    return embedded_text, time2


def train_word2vec_embedding_model(embedding_matrix, maxlen, vector_size, vocab_size):
    word_input = Input(shape=(maxlen,), dtype='float64')

    word_embedding = Embedding(input_dim=vocab_size,
                               output_dim=vector_size,
                               weights=[embedding_matrix])(word_input)

    word_vec = GlobalAveragePooling1D()(word_embedding)
    embedding_model = Model([word_input], word_vec)

    return embedding_model


def create_word2vec_embedding_matrix(model_vectors, vector_size, vocab_size, w2v_words, word_index):
    embedding_matrix = np.zeros((vocab_size, vector_size))

    i = 0
    j = 0
    for word, idx in word_index.items():
        i += 1
        if word in w2v_words:
            j += 1
            embedding_vector = model_vectors[word]
            if embedding_vector is not None:
                embedding_matrix[idx] = model_vectors[word]

    return embedding_matrix


def train_w2v_model(sentences, vector_size):
    w2v_model = gensim.models.Word2Vec(min_count=1, window=5,
                                       vector_size=vector_size,
                                       seed=42,
                                       workers=multiprocessing.cpu_count())
    w2v_model.build_vocab(sentences)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=100)

    return w2v_model


def transform_text_using_Doc2VEC(questions_without_tags):
    """Transform the text of the question's body and title into Doc2VEC embeddings."""
    time1 = time.time()

    sentences = [gensim.utils.simple_preprocess(text) for text in questions_without_tags]
    tagged_text = [TaggedDocument(words=text, tags=[str(index)]) for index, text in enumerate(sentences)]

    # dm=0 for DBOW, dm=1 for PV-DM
    model = Doc2Vec(vector_size=300, min_count=1, epochs=100, dm=0)
    model.build_vocab(tagged_text)

    model.train(tagged_text, total_examples=model.corpus_count, epochs=model.epochs)
    embedded_text = [model.infer_vector(text) for text in sentences]

    time2 = np.round(time.time() - time1, 0)
    print(f"Doc2VEC processing time:{time2}s\n")

    return embedded_text, time2


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

    return features_bert, time2


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
    return features, time2


def transform_text(questions_without_tags, text_transformation_method):
    """Transform the question text/body and title into words embeddings."""
    if text_transformation_method == "CountVectorizer":
        return transform_text_using_BagOfWords(questions_without_tags["text"], is_count_vectorizer=True)

    elif text_transformation_method == "TfidfVectorizer":
        return transform_text_using_BagOfWords(questions_without_tags["text"], is_count_vectorizer=False)

    elif text_transformation_method == "Word2Vec":
        return transform_text_using_Word2Vec(questions_without_tags["text"])

    elif text_transformation_method == "Doc2Vec":
        return transform_text_using_Doc2VEC(questions_without_tags["text"])

    elif text_transformation_method == "BERT":
        max_length = 64
        batch_size = 10
        model_type = 'bert-base-uncased'
        model = TFAutoModel.from_pretrained(model_type)

        return transform_text_using_BERT(model, model_type, questions_without_tags, max_length, batch_size)

    elif text_transformation_method == "USE":
        batch_size = 2
        return transform_text_using_USE(questions_without_tags, batch_size)


def create_results_plots(results):
    """Generate the plot showing the performances with each words embedding method for the Jaccard Score and Hamming Loss."""
    create_results_plot(results, "jaccard_score", ascending=False)
    create_results_plot(results, "hamming_loss")
    create_results_plot(results, "embedding_time")
    create_results_plot(results, "fit_time")


def create_results_plot(results, metric, ascending=True):
    results.sort_values(metric, ascending=ascending, inplace=True)

    performance_plot = (results[[metric, "words_embedding_method"]]
                        .plot(kind="bar", x="words_embedding_method", figsize=(15, 8), rot=0,
                              title=f"Models Performance Sorted by {metric}"))
    performance_plot.title.set_size(20)
    performance_plot.set(xlabel=None)

    performance_plot.get_figure().savefig(f"{RESULTS_PATH}/performance_{metric}_plot.png", bbox_inches='tight')
    plt.close()


def save_best_model(models, results_df):
    """Save the best model based on the hamming loss."""
    best_words_embedding_method = results_df.head(1)['words_embedding_method'].values[0]
    joblib.dump(models[best_words_embedding_method], f"{MODELS_PATH}/best_supervised_model.model")


def send_results_to_mlflow(best_model, hamming_loss, jaccard_score, words_embedding_method,
                           x_train, embedding_time, fit_time):
    """Send data to the MLFlow server."""
    with mlflow.start_run():
        mlflow.log_params({"words_embedding_method": words_embedding_method})

        mlflow.log_metric("hamming_loss", hamming_loss)
        mlflow.log_metric("jaccard_score", jaccard_score)
        mlflow.log_metric("embedding_time", embedding_time)
        mlflow.log_metric("fitting_time", fit_time)

        mlflow.set_tag("Words embedding method", words_embedding_method)

        signature = infer_signature(x_train, best_model.predict(x_train))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="supervised-models",
            signature=signature,
            input_example=x_train,
            registered_model_name="XGBoostClassifier",
        )


def perform_supervised_modeling(questions):
    """Find the best model using a GridSearchCV hyperoptimization for each words embedding method."""
    questions_df = DataFrame(questions)

    ml_binarizer = MultiLabelBinarizer()
    tags = ml_binarizer.fit_transform(questions_df['tags'])
    joblib.dump(ml_binarizer, f"{MODELS_PATH}/best_ml_binarizer.model")
    # questions_df['tags'].to_json(f"{MODELS_PATH}/tags.json")

    questions_without_tags = questions_df.drop(columns=['tags'], axis=1)

    results_df = DataFrame(columns=["words_embedding_method", "hamming_loss", "jaccard_score", "embedding_time",
                                    "fit_time"])
    models = {}
    for words_embedding_method in [
        "CountVectorizer",
        # "TfidfVectorizer",
        # "Word2Vec",
        # "Doc2Vec",
        # "BERT",
        # "USE"
    ]:
        print(f"Starting supervised learning of {NUMBER_OF_QUESTIONS_USED_IN_TRAINING} questions with words "
              f"embedding method:{words_embedding_method}.\n")

        transformed_text, embedding_time = transform_text(questions_without_tags, words_embedding_method)

        x_train, x_test, y_train, y_test = train_test_split(transformed_text, tags, test_size=0.2, random_state=42)
        print(f"training set size:{len(x_train)}, test set size:{len(x_test)}\n")

        # if words_embedding_method not in ("BERT", "USE"):
        #     device = "cpu"
        #     nb_jobs = -1
        # else:
        device = "cuda"
        nb_jobs = 1  # With cuda it's best to not parallelize jobs or -> cudaErrorMemoryAllocation
        x_train = cp.array(x_train)
        x_test = cp.array(x_test)

        # Best hyperparameters for 10k questions
        default_model = XGBClassifier(n_estimators=100, max_depth=5, device=device)
        default_hyperparameters = {'estimator__max_depth': range(2, 8)}

        fit_start_time = time.time()
        # grid_search_cv = GridSearchCV(MultiOutputClassifier(estimator=default_model),
        #                               default_hyperparameters,
        #                               cv=2,
        #                               scoring=make_scorer(metrics.jaccard_score, average='samples'),
        #                               n_jobs=nb_jobs,
        #                               verbose=3
        #                               )

        # grid_search_cv.fit(x_train, y_train)
        # best_parameters = grid_search_cv.best_params_
        # best_model = grid_search_cv.best_estimator_
        # print(f"\nBest Jaccard score:{grid_search_cv.best_score_} with params:{best_parameters}")

        default_model.fit(x_train, y_train)
        best_model = default_model

        fit_time = np.round(time.time() - fit_start_time, 0)

        models[words_embedding_method] = best_model

        predictions_test_y = best_model.predict(x_test)

        hamming_loss = metrics.hamming_loss(y_true=y_test, y_pred=predictions_test_y)
        jaccard_score = metrics.jaccard_score(y_true=y_test, y_pred=predictions_test_y, average='samples')
        print(f"Hamming loss:{hamming_loss}, jaccard_score:{jaccard_score}\n")

        joblib.dump(best_model, f"{MODELS_PATH}/best_supervised_model.model")

        # training set size:36000, test set size:9000, XGBClassifier
        # Hamming loss:0.0002497610080278267, jaccard_score:0.30443386243386245

        results_df.loc[len(results_df)] = [words_embedding_method, hamming_loss, jaccard_score, embedding_time,
                                           fit_time]

        # send_results_to_mlflow(best_model, hamming_loss, jaccard_score,
        #                        words_embedding_method, x_train, embedding_time, fit_time)

    create_results_plots(results_df)
    # save_best_model(models, results_df)


if __name__ == '__main__':
    print("Starting supervised learning script. Please make sure you have a local MLFlow server running.\n")
    remove_last_generated_results()

    json_questions = load_cached_questions()

    questions = [{
        "body": question['body'],
        "tags": question['tags'],
        "title": question['title']
    } for question in json_questions][:NUMBER_OF_QUESTIONS_USED_IN_TRAINING]

    print(f"{len(questions)} questions loaded from cache.\n")

    questions = list(map(extract_and_clean_text, questions))
    print("Texts extracted and cleaned.\n")

    perform_supervised_modeling(questions)

    print("\nSupervised learning now finished.\n")
