from gensim.models import KeyedVectors 
import pickle
import os

X_data = pickle.load(open('train_X_data.pkl', 'rb'))
Y_data = pickle.load(open('train_Y_data.pkl', 'rb'))

X_test = pickle.load(open('test_X_data.pkl', 'rb'))
Y_test = pickle.load(open('test_Y_data.pkl', 'rb'))

dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
word2vec_model_path = os.path.join("vi/vi.vec")

w2v = KeyedVectors.load_word2vec_format(word2vec_model_path)
vocab = w2v.wv.vocab
wv = w2v.wv

def get_word2vec_data(X):
    word2vec_data = []
    for x in X:
        sentence = []
        for word in x.split(" "):
            if word in vocab:
                sentence.append(wv[word])

        word2vec_data.append(sentence)
        print(sentence)
    return word2vec_data

X_data_w2v = get_word2vec_data(X_data)
X_test_w2v = get_word2vec_data(X_test)
