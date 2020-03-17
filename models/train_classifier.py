# import required packages
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer, TfidVectorizer
from sklearn.model_selection import train_test_split


def load_data(database_filepath):
    engine = create_engine('sqlite:///database_filepath')
    df = pd.read_sql("SELECT *  FROM InsertTableName", engine)
    X = df.iloc[:, 2]
    y = df.iloc[:,4:]
    category_names = y.columns
    
    return X, y, category_names


def tokenize(text):
    text2 = text.map(lambda x: x.lower())
    
#     cvect = CountVectorizer(stop_words='english')
    
    cvect = TfidVectorizer(stop_words='english', max_df=0.86)
    
    xcount = cvect.fit_transform(text2)
    
    df = pd.DataFrame(xcount, columns=cvect.get_feature_names())
    
    return df


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
