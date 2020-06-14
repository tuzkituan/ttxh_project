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
from keras.models import Sequential
from sklearn.decomposition import TruncatedSVD
import numpy as np
import gensim
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

X_data, y_data = get_data('train.csv')
X_test, y_test = get_data('test.csv')
X_val, y_val = get_data('valid.csv')

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

# print(encoder.classes_)

# WORD2VECTOR
# word2vec_model_path = os.path.join("vi/vi.vec")

# w2v = KeyedVectors.load_word2vec_format(word2vec_model_path)
# vocab = w2v.wv.vocab
# wv = w2v.wv

# def get_word2vec_data(X):
#     word2vec_data = []
#     for x in X:
#         sentence = []
#         for word in x.split(" "):
#             if word in vocab:
#                 sentence.append(wv[word])

#         word2vec_data.append(sentence)
#         # print(sentence)
#     return word2vec_data

# X_data_w2v = get_word2vec_data(X_data)
# X_test_w2v = get_word2vec_data(X_test)
# X_val_w2v = get_word2vec_data(X_val)

# TFIDF 
# word level - we choose max number of words equal to 30000 except all words (100k+ words)
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
tfidf_vect.fit(X_data)
# learn vocabulary and idf from training set
X_data_tfidf =  tfidf_vect.transform(X_data)
X_test_tfidf =  tfidf_vect.transform(X_test)
X_val_tfidf =  tfidf_vect.transform(X_val)

# SVD
svd = TruncatedSVD(n_components=300, random_state=42)
svd.fit(X_data_tfidf)
X_data_tfidf_svd = svd.transform(X_data_tfidf)
X_test_tfidf_svd = svd.transform(X_test_tfidf)
X_val_tfidf_svd = svd.transform(X_val_tfidf)

f = open("RESULT/EVALUATE.txt", "w")

# TRAIN MODEL
def train_model(classifier, X_data, y_data, X_test, y_test, X_val, y_val, is_neuralnet=False, n_epochs=3):       
    # X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)

    if is_neuralnet:
        classifier.fit(np.array(X_data), np.array(y_data), validation_data=(np.array(X_val), np.array(y_val)), epochs=n_epochs, batch_size=512)
        
        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)
        val_predictions = val_predictions.argmax(axis=-1)
        test_predictions = test_predictions.argmax(axis=-1)
    else:
        classifier.fit(np.array(X_data), np.array(y_data))
    
        # train_predictions = classifier.predict(X_data)
        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)
           
    # print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val))
    # print("Test accuracy: ", metrics.accuracy_score(test_predictions, y_test))

    f.write('\n- Test accuracy: ' + str(metrics.accuracy_score(test_predictions, y_test)))
    f.write('\n- Test f1-score: ' + str(metrics.f1_score(test_predictions, y_test, average='weighted')))
    # f.write('\n- Test recall: ' + str(metrics.recall_score(test_predictions, y_test, average='weighted')))
    # f.write('\n- Test precision: ' + str(metrics.precision_score(test_predictions, y_test, average='weighted')))

    f.write('\n\n- Validation accuracy: ' + str(metrics.accuracy_score(val_predictions, y_val)))
    f.write('\n- Validation f1-score: ' + str(metrics.f1_score(val_predictions, y_val, average='weighted')))
    # f.write('\n- Validation recall: ' + str(metrics.recall_score(val_predictions, y_val, average='weighted')))
    # f.write('\n- Validation precision: ' + str(metrics.precision_score(val_predictions, y_val, average='weighted')))

# LSTM
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

svmModel = svm.SVC()
f.write("[SVM + TF-IDF]")
train_model(svmModel, X_data_tfidf_svd, y_data, X_test_tfidf_svd, y_test, X_val_tfidf_svd, y_val, is_neuralnet=False)

randomForestModel = ensemble.RandomForestClassifier()
f.write("\n\n[RANDOM FOREST + TF-IDF]")
train_model(randomForestModel, X_data_tfidf_svd, y_data, X_test_tfidf_svd, y_test, X_val_tfidf_svd, y_val, is_neuralnet=False)

f.write("\n\n[LSTM + TF-IDF]")
classifier = create_lstm_model()
train_model(classifier=classifier, X_data=X_data_tfidf_svd, y_data=y_data_n, X_test=X_test_tfidf_svd, y_test=y_test_n, X_val = X_val_tfidf_svd, y_val = y_val_n, is_neuralnet=True)

# f.write("\n[LSTM + W2V]")
# classifier = create_lstm_model()
# train_model(classifier=classifier, X_data=X_data_w2v, y_data=y_data_n, X_test=X_test_w2v, y_test=y_test_n, X_val = X_val_w2v, y_val = y_val_n, is_neuralnet=True)

f.close()


def preprocessing_doc(doc):
    lines = gensim.utils.simple_preprocess(doc)
    lines = ' '.join(lines)
    lines = ViTokenizer.tokenize(lines)
    return lines

with io.open('test.txt', 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            print('================================================')
            print('Sentence: ' + line)
            test_doc = line
            test_doc = preprocessing_doc(test_doc)

            test_doc_tfidf = tfidf_vect.transform([test_doc])
            # print(np.shape(test_doc_tfidf))

            test_doc_svd = svd.transform(test_doc_tfidf)
            print("\nSVM: ")
            prediction = svmModel.predict(test_doc_svd)
            print(prediction)

            print("\nRANDOM FOREST: ")
            prediction = randomForestModel.predict(test_doc_svd)
            print(prediction)

            # print("\nLSTM: ")
            # prediction = classifier.predict(test_doc_svd)
            # print(prediction)

            print('\n')