import sys
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine 
import re 

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''
    input:
        databease_filepath = the path to database created in 
        process_data.py
    output:
        return 
    '''
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("categories", engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    categories_name = Y.columns 
   
    return X, Y, categories_name 


def tokenize(text):
    '''
    input:
        text = message to be prepared 
    output:
        clean_tokens = the messages processed 
    '''
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    #get list of urls 
    detected_urls = re.findall(url_regex, text)
    
    #replace each url 
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
   
    # sepratated in words 
    tokens = word_tokenize(text)
    
    # initiated
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens 


def build_model():
    '''
    input:
       It is not necessary 
    output:
       Model
    
    '''
    pipeline= Pipeline([
        
         ('vect', CountVectorizer(tokenizer = tokenize)),
         ('tfidf', TfidfTransformer()),
         ('clf', MultiOutputClassifier(KNeighborsClassifier()))])
    
    parameters = {'tfidf__use_idf': (True, False),
               'clf__estimator__n_neighbors' : [6,12]
                 }

    model = GridSearchCV(pipeline, param_grid = parameters)
    
    return model

def evaluate_model(model, X_test, Y_test, categories_name):
    '''
    input:
        model = model trained 
        X_test = test features 
        Y_test= test target 
        category_names = categories target 
    output:
        return the classification report 
    '''
    Y_pred = model.predict(X_test)
    report = (classification_report(Y_test, Y_pred, target_names = categories_name))
    
    print('Accuracy: {}'.format(np.mean(Y_test.values == Y_pred)))
     


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
   


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, categories_name = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, categories_name)

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