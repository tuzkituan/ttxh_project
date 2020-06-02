from pyvi import ViTokenizer, ViPosTagger
from tqdm import tqdm
from sklearn import preprocessing
import numpy as np
import gensim
import os 
import io
import pickle


def get_data(file_path):
    X = []
    Y = []
    with io.open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        sentences = []
        for line in lines:
            label = line.split(",")[0]
            sentence = line.split(",")[1]
            sentences.append(sentence)
            Y.append(label)
          
        sentences = ' '.join(sentences)
        sentences = gensim.utils.simple_preprocess(sentences)
        sentences = ' '.join(sentences)
        sentences = ViTokenizer.tokenize(sentences)
        X.append(sentences)
    return X, Y


train_X_data, train_Y_data = get_data('train_nor_811.csv')
pickle.dump(train_X_data, open('train_X_data.pkl', 'wb'))
pickle.dump(train_Y_data, open('train_Y_data.pkl', 'wb'))

test_X_data, test_Y_data = get_data('test_nor_811.csv')
pickle.dump(test_X_data, open('test_X_data.pkl', 'wb'))
pickle.dump(test_Y_data, open('test_Y_data.pkl', 'wb'))

encoder = preprocessing.LabelEncoder()
y_data_n = encoder.fit_transform(train_Y_data)
y_test_n = encoder.fit_transform(test_Y_data)

encoder.classes_