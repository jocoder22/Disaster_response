import os
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
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
)
from sklearn.pipeline import Pipeline


import plotly.graph_objects as go
import plotly


from data_wrangle2 import data_ww

sys.path.insert(0, "D:\Disaster_response")

mydir = r"D:\Disaster_response"

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
database_path = os.path.join(mydir, "data\disasterResponse.db")
print(database_path)
D:\Disaster_response\data\disasterResponse.db

engine = create_engine(f"sqlite:///{database_path}", echo=False)
df = pd.read_sql_table("disasterTable", engine)

# load model

model_path = os.path.join(mydir, "models/classifier.pkl")
model = joblib.load(f"{model_path}", "r")


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():

    # extract data needed for visuals
    figures = data_ww(df)

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(figures)]
    graphJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

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
