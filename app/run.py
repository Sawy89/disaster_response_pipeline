'''
Web app in Flask for data visualization
'''
# %% Import
import json
import plotly
import pandas as pd
import os

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import joblib
from sqlalchemy import create_engine

# %% Init & Funct
app = Flask(__name__)

def tokenize(text):
    '''
    Function for tokenize & lemmatize a text for ML pipeline
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# %% Endpoints
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    '''
    index webpage displays cool visuals and receives user input text for model
    - show some graph on the messages stored and classified
    - input form for classifying a new message
    '''
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    class_count = df.drop(columns=['id','message','original']).groupby('genre').sum()
    data2 = []
    for genre_val in class_count.index:
        tmp = class_count.loc[genre_val]
        data2.append(Bar(x=list(tmp.index),y=tmp.values, name=genre_val))
    
    # create visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres'
            }
        },

        {
            'data': data2,

            'layout': {
                'title': 'Distribution of classes',
                'barmode': 'stack',
                'yaxis': {
                    'title': "number of messages"
                },
                'xaxis': {
                    'title': "Classes",
                    'tickangle': -45,
                    'automargin': True
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    '''
    web page that handles user query and displays model results
    '''
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


# %% Main
def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()