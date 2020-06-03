from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, numpy, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from keras.layers import *

from pyvi import ViTokenizer, ViPosTagger
from tqdm import tqdm
import numpy as np
import gensim
import numpy as np

def preprocessing_doc(doc):
    lines = gensim.utils.simple_preprocess(doc)
    lines = ' '.join(lines)
    lines = ViTokenizer.tokenize(lines)

    return lines



import pickle

X_data = pickle.load(open('X_data.pkl', 'rb'))
y_data = pickle.load(open('y_data.pkl', 'rb'))

tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
tfidf_vect.fit(X_data) # learn vocabulary and idf from training set
X_data_tfidf =  tfidf_vect.transform(X_data)

encoder = preprocessing.LabelEncoder()
y_data_n = encoder.fit_transform(y_data)
print(y_data_n)
from sklearn.model_selection import train_test_split
def train_model(classifier, X_data, y_data, X_test=None, y_test=None, is_neuralnet=False, n_epochs=3):         
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
    
    if is_neuralnet:
        classifier.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, batch_size=512)
        
        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)
        val_predictions = val_predictions.argmax(axis=-1)
#         test_predictions = test_predictions.argmax(axis=-1)
    else:
        classifier.fit(X_train, y_train)
    
        train_predictions = classifier.predict(X_train)
        val_predictions = classifier.predict(X_val)
#         test_predictions = classifier.predict(X_test)
        
    print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val))
#     print("Test accuracy: ", metrics.accuracy_score(test_predictions, y_test))

model = svm.SVC()
train_model(model, X_data_tfidf, y_data, is_neuralnet=False)

test_doc = "sau này làm phò hết thôi . haizz"
test_doc = preprocessing_doc(test_doc)
test_doc_tfidf = tfidf_vect.transform([test_doc])
print(np.shape(test_doc_tfidf))
print(model.predict(test_doc_tfidf))