# !/usr/bin/python3
# coding=utf-8
from pyvi import ViTokenizer, ViPosTagger
from tqdm import tqdm
from sklearn import preprocessing, metrics, svm, linear_model, naive_bayes, ensemble
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


def get_data(file_path):
    X = []
    y = []
    with io.open(file_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            label = line.split(',')[0].encode('utf-8')
            sentence = []
            sentence.append(line.split(',')[1])
               
            y.append(label)
            sentence = ' '.join(sentence)    
            sentence = gensim.utils.simple_preprocess(sentence)
            sentence = ' '.join(sentence)
            sentence = ViTokenizer.tokenize(sentence)

            # print(lines)
            X.append(sentence)
    # print(X)
    return X, y


X_data, y_data = get_data('train_nor_811.csv')
X_test, y_test = get_data('test_nor_811.csv')
X_val, y_val = get_data('valid_nor_811.csv')

pickle.dump(X_data, open('Data/X_data.pkl', 'wb'))
pickle.dump(y_data, open('Data/y_data.pkl', 'wb'))
pickle.dump(X_test, open('Data/X_test.pkl', 'wb'))
pickle.dump(y_test, open('Data/y_test.pkl', 'wb'))
pickle.dump(X_val, open('Data/X_val.pkl', 'wb'))
pickle.dump(y_val, open('Data/y_val.pkl', 'wb'))

# CONVERT LABELS TO NUMBERS
encoder = preprocessing.LabelEncoder()
y_data_n = encoder.fit_transform(y_data)
y_test_n = encoder.fit_transform(y_test)
y_val_n = encoder.fit_transform(y_val)

# TFIDF 
# word level - we choose max number of words equal to 30000 except all words (100k+ words)
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
tfidf_vect.fit(X_data)
tfidf_vect.fit(X_test)
tfidf_vect.fit(X_val)
# learn vocabulary and idf from training set
X_data_tfidf =  tfidf_vect.transform(X_data)
X_test_tfidf =  tfidf_vect.transform(X_test)
X_val_tfidf =  tfidf_vect.transform(X_val)

# TRAIN MODEL
def train_model(classifier, X_data, y_data, X_test, y_test, X_val, y_val, is_neuralnet=False, n_epochs=3):       
    # X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)

    if is_neuralnet:
        classifier.fit(X_data, y_data, validation_data=(X_val, y_val), epochs=n_epochs, batch_size=512)
        
        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)
        val_predictions = val_predictions.argmax(axis=-1)
        test_predictions = test_predictions.argmax(axis=-1)
    else:
        classifier.fit(X_data, y_data)
    
        # train_predictions = classifier.predict(X_data)
        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)
           
    print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val))
    print("Test accuracy: ", metrics.accuracy_score(test_predictions, y_test))

print("NAIVE BAYES")
train_model(naive_bayes.MultinomialNB(), X_data_tfidf, y_data_n, X_test_tfidf, y_test_n, X_val_tfidf, y_val_n, is_neuralnet=False)
print("SVM")
train_model(svm.SVC(), X_data_tfidf, y_data_n, X_test_tfidf, y_test_n, X_val_tfidf, y_val_n, is_neuralnet=False)
print("RANDOM FOREST")
train_model(ensemble.RandomForestClassifier(), X_data_tfidf, y_data_n, X_test_tfidf, y_test_n, X_val_tfidf, y_val_n, is_neuralnet=False)

# print("LINEAR REGRESSION")
# train_model(linear_model.LogisticRegression(), X_data_tfidf, y_data, X_test_tfidf, y_test, is_neuralnet=False)

# def create_lstm_model():
#     input_layer = Input(shape=(300,))
    
#     layer = Reshape((10, 30))(input_layer)
#     layer = LSTM(128, activation='relu')(layer)
#     layer = Dense(512, activation='relu')(layer)
#     layer = Dense(512, activation='relu')(layer)
#     layer = Dense(128, activation='relu')(layer)
    
#     output_layer = Dense(10, activation='softmax')(layer)
    
#     classifier = models.Model(input_layer, output_layer)
    
#     classifier.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
#     return classifier

# classifier = create_lstm_model()
# train_model(classifier=classifier, X_data=X_data_tfidf, y_data=y_data_n, X_test=X_test_tfidf, y_test=y_test_n, is_neuralnet=True)
