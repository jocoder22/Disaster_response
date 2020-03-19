# import required packages
import sys
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.linear_model import LogisticRegression  
from sklearn.multiclass import OneVsRestClassifier 
from sklearn.ensemble import RandomForestClassifier


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, multilabel_confusion_matrix
from sklearn.metrics import hamming_loss


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
    engine = create_engine(f"sqlite:///{database_filepath}", echo=False)

    # read all data in sql table
    df =  pd.read_sql_query("SELECT  *  FROM  disasterTable", engine)

    # Select text and target
    messages = df.iloc[:, 1].values
    categories = df.iloc[:, 4:].values

    # get categories names
    category_names = df.iloc[:, 4:].columns

    # tokenize text
    # X_tokenized = tokenize(X_raw)

    return messages, categories, category_names


def tokenize(text):
    """The tokenize function will form tokenization for the text messages
        to use for model training and testing

    Args:
        text (DataFrame): the DataFrame with text column for tokenization


    Returns:
        DataFrame: The DataFrame with words tokens and values for modelling

    """
    # normalize the text to all lower case
    text2 = text.apply(lambda x: x.lower())

    #  cvect = CountVectorizer(stop_words='english')

    # create a TfidfVectorizer
    cvect = TfidfVectorizer(stop_words="english", max_df=0.86)

    # Fit and transform text
    xcount = cvect.fit_transform(text2)

    # Create pandas DataFrame
    df = pd.DataFrame(xcount.A, columns=cvect.get_feature_names())

    return df


def build_model():
    """The build_model function build a model pipeline

    Args: None

    Returns:
        model(pipeline): model pipeline for fitting, prediction and scoring

    """
    # create pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("naiveclass", MultinomialNB())
    ])

    pipe2 = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", OneVsRestClassifier(LogisticRegression()))
    ])

    # pl = Pipeline([
    #     ('vec', CountVectorizer()),
    #     ('clf', OneVsRestClassifier(LogisticRegression()))
    # ])
    pl = Pipeline([
        ('vec', CountVectorizer()),
        ("forestClass", RandomForestClassifier())
    ])

    model1 = OneVsRestClassifier(LogisticRegression())
    
    return pl

    # return pipe2


def evaluate_model(model, X_text, Y_test, category_name):
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
    # predict using the model
    # pred = model.predict(X_text)


    # calculate the accuracy score
    # a_score = hamming_loss(Y_test, pred)
    # a_score = metrics.accuracy_score(Y_test, pred, normalize=True, sample_weight=None)
   
    # a_score = model.score(X_text, Y_test)

    # cm = multilabel_confusion_matrix(y_test, ypred)
    # calculate the confusion matrix
    # conf_mat = multilabel_confusion_matrix(Y_test, pred, labels=category_name)
    # print(f"Confusion Matrix:\n{conf_mat}\n")


    # print(f"Model Accuracy: { (1-a_score)*100:.02f}%\n\n")
    print("pass... evaluation")

    


def save_model(model, model_filepath):
    """The save_model function save the model

    Args:
        model (model): the model to save
        model_filepath (filepath): filepath where to save the modeld



    Returns: None
            print out: Done saving model!

    """

    # # Save the model

    pickle.dump(model, open(model_filepath, 'wb'))

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
