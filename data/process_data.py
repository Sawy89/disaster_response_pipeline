'''
Script for taking CSV data, loading & cleaning them, and save in DB

INPUT:
    messages_filepath = 'disaster_messages.csv'
    categories_filepath = 'disaster_categories.csv'

    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
'''

# %% Import 
import sys
import pandas as pd
import os
from sqlalchemy import create_engine


# %% Functions
def load_data(messages_filepath, categories_filepath):
    '''
    Load CSV files with data and merge dataframes
    '''

    # Load
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)

    # Reset index & Join
    df_messages.set_index(['id'], inplace=True)
    df_categories.set_index(['id'], inplace=True)

    df = df_messages.join(df_categories)

    return df


def clean_data(df):
    '''
    Clean the DataFrame:
    - categories columns is split in many columns and parsed as bolean
    '''

    # Clean categories column
    categories_list = [i.split('-')[0] for i in df['categories'].iloc[0].split(';')]
    categories = df['categories'].str.split(';', expand=True)
    categories.columns = categories_list
    categories = categories.applymap(lambda x: int(x.split('-')[1]))
    
    # join
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)

    # Drop duplicates & strange values
    df = df.drop_duplicates()
    df = df.loc[(df==2).sum(axis=1)==0]

    return df


def save_data(df, database_filename):
    '''
    upload df data on SQLlite database
    Replace if exists
    '''

    # Create engine & upoad
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages', engine, if_exists='replace')


# %% Main
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()