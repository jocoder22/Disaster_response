# import required packages
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer, TfidVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


def load_data(database_filepath):
    """The load_data function

    Args:
        database_filepath (filepath): the sql database filepath


    Returns:
        DataFrame: The DataFrame for analysis
        X_tokenized (DataFrame): dataframe with text message column
        y (DataFrame):  dataframe with target classes
        category_names(list): list containing the name of the categories

    """
    # create sql engine
    engine = create_engine("sqlite:///database_filepath")

    # read all data in sql table
    df = pd.read_sql("SELECT *  FROM InsertTableName", engine)

    # Select text and target
    X_raw = df.iloc[:, 2]
    y = df.iloc[:, 4:]
    category_names = y.columns

    # tokenize text
    X_tokenized = tokenize(X)

    return X_tokenized, y, category_names



def tokenize(text):
    """The tokenize function will form tokenization for the text messages
        to use for model training and testing

    Args:
        text (DataFrame): the DataFrame with text column for tokenization


    Returns:
        DataFrame: The DataFrame with words tokens and values for modelling

    """
    # normalize the text to all lower case
    text2 = text.map(lambda x: x.lower())

    #  cvect = CountVectorizer(stop_words='english')

    # create a TdifVectorizer
    cvect = TfidVectorizer(stop_words="english", max_df=0.86)

    # Fit and transform text
    xcount = cvect.fit_transform(text2)

    # Create pandas DataFrame
    df = pd.DataFrame(xcount, columns=cvect.get_feature_names())

    return df



def build_model():
    """The build_model function build a model pipeline
 
    Args: None
 
    Returns: 
        model(pipeline): model pipeline for fitting, prediction and scoring
 
    """
    pass
    


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


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
