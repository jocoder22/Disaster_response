import os
import sys
import json
import plotly
import pandas as pd
import pickle
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar

# from sklearn.externals import joblib
from sqlalchemy import create_engine

from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
)
from sklearn.pipeline import Pipeline


import plotly.graph_objects as go
import plotly

from data_wrangle2 import data_ww

sys.path.insert(0, "D:/Disaster_response/app")
sys.path.insert(1, "D:/Disaster_response/models")
mydir = r"D:\Disaster_response"


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    wordporter = SnowballStemmer("english")

    stopword = set(stopwords.words("english"))
    
    # Retain alphabetic words: alpha_only
    alpha_only = [t.lower() for t in tokens if t.isalpha()]

    # Remove all stop words: no_stops
    _tokens = [t for t in alpha_only if t not in stopword]

    no_stop_tokens = [wordporter.stem(word) for word in _tokens]

    clean_tokens = []

    # for tok in tokens:
    for tok in no_stop_tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


path = r"D:\Disaster_response\data"
os.chdir(path)

# Create engine: engine
engine = create_engine("sqlite:///disasterResponse.db", echo=False)


# load data
# engine = create_engine(f"sqlite:///{database_path}", echo=False)
df = pd.read_sql_table("disasterTable", engine)

# load model
model_path = "../models/classifier.pkl"
# model_path = os.path.join(mydir, "models/classifier.pkl")
model = joblib.load(f"{model_path}", "r")


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():

    # extract data needed for visuals
    fig = data_ww(df)

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(fig)]
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(
        zip(df.columns[4:], classification_labels)
    )

    # This will render the go.html Please see that file.
    return render_template(
        "go.html",
        query=query,
        classification_result=classification_results,
    )


def main():
    # app.run(host='0.0.0.0', port=3001, debug=True)
    app.run(host="127.0.0.1", port=8080, debug=True)


if __name__ == "__main__":
    main()
