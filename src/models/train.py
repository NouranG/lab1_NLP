#importing necessary libraries
import json
import yaml
import joblib
import pandas as pd
import mlflow
import numpy as np
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

#importing scripts and modules
from src.preprocessing.textpreprocessing import PreprocessingConfig, TextPreprocessor
from src.vectorization.embeddings import Embeddings
from src.vectorization.bow import BoWVectorizer
from src.vectorization.tfidf import TFIDFVectorizer
from src.vectorization.bm25 import BM25Vectorizer



# Load parameters from YAML file
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "src" / "params.yaml"

with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)

#-----------------------------------------------------------------------------

# Load data
data_path = ROOT / params["data"]["input_path"]
df = pd.read_csv(data_path)

text_column = params["data"]["text_column"]
label_column = params["data"]["label_column"]
df= df[[text_column, label_column]]

df =df.dropna(subset=[text_column, label_column])
# changing label into binary feature
df['label'] = df[label_column].apply(lambda x: 1 if x >= 4 else 0)

#---------------------------------------------------------------------------------

# Preprocess text
preprocessing_domain = params["preprocessing"]["domain"]

if preprocessing_domain == "twitter":

    config = PreprocessingConfig.for_tweets()

elif preprocessing_domain == "reviews":

    config = PreprocessingConfig.for_reviews()

else:
    config = PreprocessingConfig()

preprocessor = TextPreprocessor(config)
df['clean_text'] = df[text_column].apply(preprocessor.process)

#---------------------------------------------------------------------------------

#train test split
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], 
                            test_size=params["model"]["test_size"]
                            , random_state=params["model"]["random_state"])

#-----------------------------------------------------------------------------
# Vectorization
#1. BoW
vec_type = params["vectorization"]["type"]

if vec_type == "bow":
    bow_vectorizer = BoWVectorizer(preprocessor, max_features=params["vectorization"]["max_features"])
    X_train_vec = bow_vectorizer.fit_transform(X_train)
    X_test_vec = bow_vectorizer.transform(X_test)
#2. TF-IDF
elif vec_type == "tfidf":
    tfidf_vectorizer = TFIDFVectorizer(preprocessor, max_features=params["vectorization"]["max_features"])
    X_train_vec = tfidf_vectorizer.fit_transform(X_train)
    X_test_vec = tfidf_vectorizer.transform(X_test)
#3. BM25
elif vec_type == "bm25":
    bm25_vectorizer = BM25Vectorizer(max_features=params["vectorization"]["max_features"])
    X_train_vec = bm25_vectorizer.fit_transform(X_train)
    X_test_vec = bm25_vectorizer.transform(X_test)

#4. Embeddings
elif vec_type == "embeddings":
    emb_params = params["vectorization"]["embeddings"]
    vectorizer = Embeddings(
        embedding_type=emb_params["type"],
        embedding_path=emb_params["embedding_path"],
        vector_size=emb_params["vector_size"],
        window=emb_params["window"],
        min_count=emb_params["min_count"],
    workers=emb_params["workers"]
)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

#---------------------------------------------------------------------------------
#mlflow start run
with mlflow.start_run():
    
#---------------------------------------------------------------------------------
# Model Training
    if params["model"]["type"] == "random_forest":
        model = RandomForestClassifier(random_state=params["model"]["random_state"])
    elif params["model"]["type"] == "naive_bayes":
        model = GaussianNB()
    elif params["model"]["type"] == "logistic_regression":
        model = LogisticRegression(random_state=params["model"]["random_state"])
    else:
        raise ValueError("Unsupported model type")

    #predicting and evaluating the model
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")


    #---------------------------------------------------------------------------------
    #mlflow logging
    mlflow.log_param("vectorizer", vec_type)
    mlflow.log_param("model", params["model"]["type"])

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    mlflow.sklearn.log_model(model, "model")