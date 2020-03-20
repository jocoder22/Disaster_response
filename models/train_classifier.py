import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, FunctionTransformer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import PolynomialFeatures

import nltk
# from nltk import ConfusionMatrix
from nltk.metrics import ConfusionMatrix
nltk.download(['punkt','wordnet', 'averaged_perceptron_tagger'])

from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import joblib


#  custom transformer
class dummyTransformer(BaseEstimator, TransformerMixin):
    """dummyTransformer class forms dummies from selected columns"""
#     def __init__(self, prefix_separator="_", columnlist=['unrelated', 'genre']):
        
#         self.prefix_separator = prefix_separator
#         self.columnlist = columnlist

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        finaldata = pd.get_dummies(X)
        
        return finaldata

    


# custom transformer
class colSelector(BaseEstimator, TransformerMixin):
    """dummyTransformer class forms dummies from selected columns"""
    def __init__(self, col=0):
        
        self.columnlist = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        col_ = X[:, self.columnlist]
        
        return col_
    
    
def load_data(database_filepath):
    """The load_data function loads the dataset 
       from sql database.

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
    df = pd.read_sql_query("SELECT  *  FROM  disasterTable", engine)

    # Select text and target
    messages_ = df.iloc[:, 1].values
    categories_ = df.iloc[:, 4:].values

    # get categories names
    category_names = df.iloc[:, 4:].columns

    return messages_, categories_, category_names


def tokenize(text):
    """The tokenize function will form tokenization for the text messages
        to use for model training and testing

    Args:
        text (text message): the text message for tokenization


    Returns:
        token_cleaned (list): list with words tokens for modelling

    """
    tokens = word_tokenize(text)
    lemmy = WordNetLemmatizer()

    token_cleaned = []

    for token in tokens:
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
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("naiveclass", MultinomialNB())]
    )

    pipe2 = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", OneVsRestClassifier(LogisticRegression())),
        ]
    )

    # pl = Pipeline([
    #     ('vec', CountVectorizer()),
    #     ('clf', OneVsRestClassifier(LogisticRegression()))
    # ])
    pl = Pipeline(
        [
            ("vec", CountVectorizer()),
            ("forestClass", RandomForestClassifier()),
        ]
    )

    model1 = OneVsRestClassifier(LogisticRegression())

    pll = Pipeline([
            ('cvect', CountVectorizer(tokenizer=tokenize)),
            ('tfidt', TfidfTransformer()),
            #     ('rforest', RandomForestClassifier()),
            ('multi', MultiOutputClassifier(KNeighborsClassifier()))
            ])

    return pll



def evaluate_model(model, X_test, Y_test, category_name):
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
    pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = (pred == Y_test).mean()
    accuracyscore = model.score(X_test, Y_test)


    for i, label in enumerate(category_name):
        print("Printing for ", label)
        print(classification_report(Y_test[i] , pred[i]), **sp)

    print(f"Model Accuracy: {accuracy*100:.02f}%\n")
    print(f"Model Accuracy: {accuracyscore*100:.02f}%\n")
    


def save_model(model, model_filepath):
    """The save_model function save the model

    Args:
        model (model): the model to save
        model_filepath (filepath): filepath where to save the modeld



    Returns: None
            print out: Done saving model!

    """

    # # Save the model

    pickle.dump(model, open(model_filepath, "wb"))

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
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
