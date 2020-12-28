'''
database_filepath = 'data/disaster_data.db'
model_filepath = 'models/model.pkl'

python models/train_classifier.py data/disaster_data.db models/model.pkl
'''

# %% Download nltk
import nltk
nltk.download(['punkt', 'wordnet'])


# %% Import
import sys
import datetime
import re
import joblib
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix


# %% Functions
def load_data(database_filepath):
    ''' 
    Function that loads DB database_filepath - table messages
    Return the table diveded in X, y and category_names
    '''
    # Load data
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('SELECT * FROM messages', engine)

    # Get output variables
    df.set_index('id', inplace=True)
    df.drop(columns=['original'], inplace=True)
    X = df['message']
    Y = df.drop(columns=['message','genre'])
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    # Convert to lowercase
    text = text.lower() 

    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 

    # tokenize
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Report the f1 score, precision and recall for each output category of the dataset. 
    You can do this by iterating through the columns and calling sklearn's classification_report on each.
    '''
    # Y_pred = model.predict(X_test)
    # confusion_mat = confusion_matrix(Y_test, Y_pred, labels=category_names)
    # accuracy = (Y_pred == Y_test).mean()


def save_model(model, model_filepath):
    joblib.dump(model,  open(model_filepath, 'wb'))
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        start = datetime.datetime.now()
        model.fit(X_train, Y_train)
        print(f'   DONE in {datetime.datetime.now()-start}')
        
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


# %% Main
if __name__ == '__main__':
    main()
