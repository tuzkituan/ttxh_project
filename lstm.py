# -*- coding: utf-8 -*-
import numpy as np
import sys
import codecs


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.preprocessing import sequence

# Make (text, index) dictionary.
def make_text_index_dic(_text):
	word_set = set()
	for text in _text:
		for word in text:
			word_set.add(word)

	word_dic = {'UNK_WORD':0}
	i = 1
	for word in word_set:
		word_dic.update({word:i})
		i = i + 1
	return word_dic

# Map text list to index list.
def map_text_to_index(_text, _dic, max_rnn_len):
	x_index = []
	x_element = []
	for text in _text:
		for word in text:
			if word in _dic:
				x_element.append(_dic.get(word))
			else:
				x_element.append(_dic.get('UNK_WORD'))
		x_index.append(x_element)
		x_element = []
	x_index = sequence.pad_sequences(x_index, maxlen=max_rnn_len)
	return x_index

# Map label list to index list.
def map_label_to_index(_labels):
	label_dic = {
		'Disgust':[1,0,0,0,0,0,0],
		'Enjoyment':[0,1,0,0,0,0,0],
		'Sadness':[0,0,1,0,0,0,0],
		'Surprise':[0,0,0,1,0,0,0],
		'Anger':[0,0,0,0,1,0,0],
		'Fear':[0,0,0,0,0,1,0],
		'Other':[0,0,0,0,0,0,1]}
	y_index = []
	for label in _labels:
		y_index.append(label_dic[label])
	return np.array(y_index, dtype=np.int)


# Read train data.
train_text = []; train_labels = []
for line in codecs.open('train.csv', 'r', 'utf-8'): 
	label, text = line.strip().split(',')
	train_text.append(text)
	train_labels.append(label)


# Read test data
test_text = []; test_labels = []
for line in codecs.open('test.csv', 'r', 'utf-8'):
	label, text = line.strip().split(',')
	test_text.append(text)
	test_labels.append(label)

f = open('result_lstm.txt', 'w')
f.write('[LSTM]\n')

max_features = 128
MAX_RNN_LEN = 50

total_acc = 0.0

# TRAIN AND TEST
print('====== TEST =====\n')
# test_labels, test_text, _labels, _text = select_test_data(sample_labels, sample_text, i)	
# train_labels = base_labels + _labels
# train_text = base_text + _text

text_index_dic = make_text_index_dic(train_text + test_text)
x_train = map_text_to_index(train_text, text_index_dic, MAX_RNN_LEN)
y_train = map_label_to_index(train_labels)
x_test = map_text_to_index(test_text, text_index_dic, MAX_RNN_LEN)
y_test = map_label_to_index(test_labels)

model = Sequential()
model.add(Embedding(len(text_index_dic), output_dim=128, input_length=MAX_RNN_LEN, mask_zero=True))
model.add(LSTM(128))
#model.add(Dropout(0.5))
model.add(Dense(7))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', 
		optimizer='adam',
		metrics=['accuracy'])
		
print('Fitting...')
model.fit(x_train, y_train, batch_size=32, epochs=5)
print('Done')

y_pred = model.predict(x_test)
y_pred = [np.argmax(_) for _ in y_pred]
y_true = [np.argmax(_) for _ in y_test]
f.write(' - Prediction = ' + str(y_pred) + '\n')
f.write(' - True = ' + str(y_true) + '\n')	
accuracy = np.average([y_pred[i] == y_true[i] for i in range(len(y_pred))])
f.write(' - Accuracy = ' + str(accuracy) + '\n')

total_acc += accuracy

f.write('\n* Total Accuracy = ' + str(total_acc))	
f.close()
