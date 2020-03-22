# import required packages
import sys
import numpy as np
import pandas as pd
import pickle
import joblib
from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer



# custom transformer
class dummyTransformer(BaseEstimator, TransformerMixin):
    """dummyTransformer class forms dummies from selected columns"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        finaldata = pd.get_dummies(X)
        
        return finaldata

    


# custom transformer
class columnSelector(BaseEstimator, TransformerMixin):
    """columnSelector class select columns"""
    def __init__(self, col=0):
        
        self.columnlist = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        col_ = X[:, self.columnlist]
        
        return col_
    


def load_data(database_filepath):
    """The load_data function

    Args:
        database_filepath (filepath): the sql database filepath

    Returns:
        X_tokenized (DataFrame): dataframe with text message column
        y (DataFrame):  dataframe with target classes
        category_names(list): list containing the name of the categories

    """
    # create sql engine
    engine = create_engine(f"sqlite:///{database_filepath}", echo=False)
    

    # read all data in sql table
    df =  pd.read_sql_table('disasterTable', engine)

    # drop duplicates and original text message
    df.drop_duplicates(subset = ["message"], keep="first", inplace=True)

    # drop nan, na
    df.dropna(inplace=True)

    # Select text and target
    messages_ = df.iloc[:, 0].values

    # get categories names
    categories_ = df.iloc[:, 2:].values
    name = df.iloc[:, 2:]

    # get categories names
    category_names = name.columns.tolist()

    return messages_, categories_, category_names


def tokenize(text):
    """The tokenize function will form tokenization for the text messages
        to use for model training and testing

    Args:
        text (DataFrame): the DataFrame with text column for tokenization


    Returns:
        DataFrame: The DataFrame with words tokens and values for modelling

    """

    tokens = word_tokenize(text)
    lemmy = WordNetLemmatizer()

    stopword = set(stopwords.words("english"))
    
    # Retain alphabetic words: alpha_only
    alpha_only = [t.lower() for t in tokens if t.isalpha()]

    # Remove all stop words: no_stops
    no_stop_tokens = [t for t in alpha_only if t not in stopword]

    token_cleaned = []

    for token in no_stop_tokens:
    # for token in tokens:
        token_ = lemmy.lemmatize(token).lower().strip()
        token_cleaned.append(token_)

    return token_cleaned



def build_model():
    """The build_model function build a model pipeline

    Args: None

    Returns:
        model(pipeline): model pipeline for fitting, prediction and scoring

    """
    # create pipeline
    plu = Pipeline([
                    ('cvect', CountVectorizer(tokenizer=tokenize,
                                 max_df=0.86, ngram_range=(1,2))),
                      ('tfidt', TfidfTransformer()),
                     ("mascaler", MaxAbsScaler()),
                     ('rforest', MultiOutputClassifier(RandomForestClassifier()))
            ])


    return plu


def evaluate_model(model, X_text, Y_test, category_names):
    """The evaluate_model function scores the performance of trained model
        on test (unseen) text and categories

    Args:
        model (model): model to evaluate
        X_text (numpy arrays): the test (unseen) tokenized text
        Y_test (numpy arrays): the test (unseen) target used for evaluation
        category_names(list): list containing the name of the categories

    Returns: None
            print out the accuracy and confusion metrics

    """
    sp = {"end": "\n\n", "sep": "\n\n"}

    # predict using the model
    pred = model.predict(X_text)

    # Calculate accuracy
    accuracy = (pred == Y_test).mean()
    accuracyscore = model.score(X_text, Y_test)
    
    
    print(f"Model Accuracy: {accuracy*100:.02f}%\n")
    print(f"Model Accuracy: {accuracyscore*100:.02f}%\n")


    for i, label in enumerate(category_names):
        print("Printing for ", label)
        print(classification_report(Y_test[i] , pred[i]), **sp)



def save_model(model, model_filepath):
    """The save_model function save the model

    Args:
        model (model): the model to save
        model_filepath (filepath): filepath where to save the modeld

    Returns: None
            print out: Done saving model!

    """
    # # Save the model
    joblib.dump(model, f"{model_filepath}")

    print("Done saving model!")


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(
            "Loading data...\n    DATABASE: {}".format(database_filepath)
        )
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2
        )

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl")


if __name__ == "__main__":
    main()
