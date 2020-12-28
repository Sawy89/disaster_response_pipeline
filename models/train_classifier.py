'''
messages_filepath = 'disaster_messages.csv'
categories_filepath = 'disaster_categories.csv'

database_filepath = 'data/disaster_data.db'

python models/train_classifier.py
'''

# %% Import
import sys
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import pandas as pd


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
    X = df[['message','genre']]
    Y = df.drop(columns=['message','genre'])
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    pass


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


# %% Main
if __name__ == '__main__':
    main()
