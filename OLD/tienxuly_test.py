# !/usr/bin/python3
# coding=utf-8
from pyvi import ViTokenizer, ViPosTagger
from tqdm import tqdm
from sklearn import preprocessing, metrics, svm, linear_model, naive_bayes
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.layers import *
from keras import layers, models, optimizers
import numpy as np
import gensim
import os 
import io
import pickle

dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
dir_path = os.path.join(dir_path, 'new_data')

def get_data(folder_path):
    X = []
    y = []
    dirs = os.listdir(folder_path)
    for path in tqdm(dirs):
        file_paths = os.listdir(os.path.join(folder_path, path))
        for file_path in tqdm(file_paths):
            with io.open(os.path.join(folder_path, path, file_path), 'r', encoding="utf-8") as f:
                lines = f.readlines()
                   
                lines = ' '.join(lines)
                lines = gensim.utils.simple_preprocess(lines)
                print(lines)
                lines = ' '.join(lines)
               
                lines = ViTokenizer.tokenize(lines)
                
                X.append(lines)
              
                y.append(path)  
    print(X)       
    return X, y

X_data, y_data = get_data('new_data')
# print(X_data)
X_test = X_data
y_test = y_data

# CONVERT LABELS TO NUMBERS
encoder = preprocessing.LabelEncoder()
y_data_n = encoder.fit_transform(y_data)
y_test_n = encoder.fit_transform(y_test)
# print(encoder.classes_)


# TFIDF 
# word level - we choose max number of words equal to 30000 except all words (100k+ words)
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
tfidf_vect.fit(X_data) # learn vocabulary and idf from training set
X_data_tfidf =  tfidf_vect.transform(X_data)
# assume that we don't have test set before
X_test_tfidf =  tfidf_vect.transform(X_test)
# print(X_data_tfidf)

# TRAIN MODEL
def train_model(classifier, X_data, y_data, X_test, y_test, is_neuralnet=False, n_epochs=3):       
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
    if is_neuralnet:
        classifier.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, batch_size=512)
        
        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)
        val_predictions = val_predictions.argmax(axis=-1)
        test_predictions = test_predictions.argmax(axis=-1)
    else:
        classifier.fit(X_train, y_train)
    
        train_predictions = classifier.predict(X_train)
        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)

    print("Train accuracy: ", metrics.accuracy_score(train_predictions, y_train))    
    print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val))
    print("Test accuracy: ", metrics.accuracy_score(test_predictions, y_test))


# train_model(naive_bayes.MultinomialNB(), X_data_tfidf, y_data_n, X_test_tfidf, y_test_n, is_neuralnet=False)


def create_lstm_model():
    input_layer = Input(shape=(300,))
    
    layer = Reshape((10, 30))(input_layer)
    layer = LSTM(128, activation='relu')(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dense(128, activation='relu')(layer)
    
    output_layer = Dense(10, activation='softmax')(layer)
    
    classifier = models.Model(input_layer, output_layer)
    
    classifier.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return classifier

classifier = create_lstm_model()
train_model(classifier=classifier, X_data=X_data_tfidf, y_data=y_data_n, X_test=X_test_tfidf, y_test=y_test_n, is_neuralnet=True)
