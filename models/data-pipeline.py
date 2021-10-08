# import packages
import sys
import re
import numpy as np
import pandas as pd
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier

from sqlalchemy import create_engine
engine = create_engine('sqlite:///DisasterResponseData.db')

def load_data(data_file):
    # read in file
    df = pd.read_sql_table('Data',engine)
    print(df.head())
    # clean data
    df.drop(columns=['id', 'original', 'genre'], inplace=True)
    df.dropna(inplace=True)

    # load to database
    df.to_sql('Clean_Data', engine, if_exists='replace')

    # define features and label arrays
    X = df['message'].values
    y = df.iloc[:,1:].values
    category_names = df.iloc[:,1:].columns

    return X, y, category_names

# helper function to tokenize data
def tokenize(text):
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]

    # Lemmatization - Reduce words to their root form 
    tokens = [WordNetLemmatizer().lemmatize(tok) for tok in tokens]
    
    # Stemming - Reduce words to their stems
    clean_tokens = [PorterStemmer().stem(tok) for tok in tokens] 
    
    return clean_tokens

def evaluate_model(model, X_test, Y_test, category_names):
    print("Best parameter (CV score=%0.3f):" % model.best_score_)
    print(model.best_params_)
    y_pred = model.predict(X_test)
    for i, c in enumerate(category_names):
        print(classification_report(Y_test[i], y_pred[i], target_names = category_names))  

def build_model(category_names):
    # text processing and model pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
    
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
    
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # define parameters for GridSearchCV
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, parameters)
    evaluate_model(model_pipeline, X_test, Y_test, category_names)

    return model_pipeline


def train(X, y, model):
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # fit model
    model.fit(X_train, y_train)
    
    # output model test results
    y_pred = model.predict(X_test)

    return model


def export_model(model):
    # Export model as a pickle file
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))


def run_pipeline(data_file):
    X, y, category_names = load_data(data_file)  # run ETL pipeline
    model = build_model(category_names)  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model)  # save model


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline

