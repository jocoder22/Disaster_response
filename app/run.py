import sys
import json
import plotly
import pandas as pd
import pickle
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
from sqlalchemy import create_engine

from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline



sys.path.insert(1, 'D:\Disaster_response2\models')
sys.path.insert(2, 'D:\Disaster_response2\app')
from train_classifier2 import tokenize


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
<<<<<<< HEAD
database_path = "data/DisasterResponse.db"
=======
<<<<<<< HEAD
database_path = "D:/Disaster_response2/data/DisasterResponse.db"
=======
<<<<<<< HEAD
database_path = "D:/Disaster_response2/data/DisasterResponse.db"
=======
database_path = "data/DisasterResponse.db"
>>>>>>> a57d9af7c79664c0323f32f5fac3ae23c00606ba
>>>>>>> eee255ada9eddf0cb8057930709b318d8a4d5262
>>>>>>> fc5f29392dcef867646c760353a56392c7e8847e
engine = create_engine(f"sqlite:///{database_path}", echo=False)
df = pd.read_sql_table('disasterTable', engine)

# load model
<<<<<<< HEAD
model_path = "models/classifier.pkl"
=======
<<<<<<< HEAD
model_path = "D:/Disaster_response2/models/classifier.pkl"
=======
<<<<<<< HEAD
model_path = "D:/Disaster_response2/models/classifier.pkl"
=======
model_path = "models/classifier.pkl"
>>>>>>> a57d9af7c79664c0323f32f5fac3ae23c00606ba
>>>>>>> eee255ada9eddf0cb8057930709b318d8a4d5262
>>>>>>> fc5f29392dcef867646c760353a56392c7e8847e
model = joblib.load(f"{model_path}", "r")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    # return render_template(path_master, ids=ids, graphJSON=graphJSON)
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    # app.run(host='0.0.0.0', port=3001, debug=True)
    app.run(host='127.0.0.1', port=8080, debug=True)


if __name__ == '__main__':
    main()
