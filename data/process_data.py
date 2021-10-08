import sys
import pandas as pd
from sqlalchemy import create_engine
# engine_name = 'sqlite:///'+ database_filepath
engine = create_engine('sqlite:///DisasterResponseData.db')
# engine = create_engine(engine_name)


def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath, dtype=str)
    # load categories dataset
    categories = pd.read_csv(categories_filepath, dtype=str)

    # merge datasets
    df = messages.merge(categories, how='outer', on=['id'])

    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';', expand=True)

    # Use the first row in categories 
    row = categories.iloc[0]
    row = row.str.split('-', expand=True)

    # use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x:x[0] , axis=1)
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] =categories[column].str.split('-', expand=True)[1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Drop the old categories column from data
    df.drop(['categories'],axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df , categories], join='outer' , sort=False , axis=1)   

    return df


def clean_data(df , debug=False):
    # check number of duplicates
    if df.duplicated().sum() > 0:
        # drop duplicates
        df = df.drop_duplicates()

    if debug:
        # DEBUG
        print('dataset is ready:')
        print(df.head(5))
        print()

    return df


def save_data(df, database_filename):

    df.to_sql('Data', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df , debug=True)

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