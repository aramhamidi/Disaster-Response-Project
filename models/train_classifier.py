import sys
import re
import numpy as np
import pandas as pd
import pickle

import nltk
# nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
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

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def load_data(database_filepath):
    # read in file
    df = pd.read_sql_table('Data',engine)

    # clean data
    df.drop(columns=['id', 'original', 'genre'], inplace=True)
    df.dropna(inplace=True)

    # load to database
    df.to_sql('Clean_Data', engine, if_exists='replace')

    # define features and label arrays
    X = df['message'].values
    y = df.iloc[:,1:].values
    category_names = df.iloc[:, 1:].columns
    
    return X, y, category_names


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


def build_model():
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
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 3]
    }

    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, parameters)

    return model_pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    print("Best parameter (CV score=%0.3f):" % model.best_score_)
    print(model.best_params_)
    y_pred = model.predict(X_test)
    for i, c in enumerate(category_names):
        print(classification_report(Y_test[i], y_pred[i])) 


def save_model(model, model_filepath):
    # Export model as a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


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