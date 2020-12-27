# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import sys 

import nltk
nltk.download(["punkt", "wordnet", "averaged_perceptron_tagger", "stopwords"])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    Load data from sqlite database

    Arguments:
        database_filepath (string): path to sqlite database file

    Returns:
        X : message of disaster response
        Y : categories of the message
    """

    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table("disaster_response", engine)
    X = df["message"]
    Y = df.iloc[:,4:]

    return X, Y

def tokenize(text):
    """
    Tokenize the input message for model build in next steps

    Arguments:
        text (string): input message

    Returns:
        clean_words (list): list of words 
    """
    # Normalize
    normalized_text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize
    words = word_tokenize(normalized_text)
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    clean_words = [lemmatizer.lemmatize(w).strip() for w in words]
    clean_words = [lemmatizer.lemmatize(w, pos="v").strip() for w in clean_words]
    
    return clean_words

def build_model():
    """
    Build the model

    Arguments: None
    Returns:
        cv : the constructed pipeline
    """
    # Build a pipeline
    pipeline = Pipeline([
        ("vect", CountVectorizer(tokenize)),
        ("best", TruncatedSVD()),
        ("tfidf", TfidfTransformer()),
        ("clf", MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # Hypertuning
    parameters = {
        "clf__estimator__n_estimators": [50, 100],
        "clf__estimator__min_samples_split": [2, 4]
    }

    # Cross validation
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model by reporting the f1 score, precision and recall for each output category of the dataset

    Arguments:
        model: constructed model
        X_test: input message
        y_test: output category class

    Return:
        Print out the classification report result
    """

    y_pred = model.predict(X_test)
    for i, col in enumerate(y_test):
        print(col)
        print(classification_report(y_test[col], y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Store the model to target directory

    Arguments:
        model: input model
        model_filepath: location to store the model

    Returns:
        Store the model to specific location
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print("Load data ...")
        X, Y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        print("Build model ...")
        model = build_model()

        print("Train model ...")
        model.fit(X_train, y_train)

        print("Evaluate model ...")
        evaluate_model(model, X_test, y_test)

        print("Save model ...")
        save_model(model, model_filepath)

        print("Modeling process completed successfully!")

    else:
        print("""
        Please provide the database_filepath and model_filepath correctly as below example:
        python train_classifier.py ./../data/disaster_response.db classifier.pkl
        """)

if __name__=="__main__":
    main()