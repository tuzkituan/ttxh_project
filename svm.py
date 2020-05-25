# -*- coding: utf-8 -*-
import numpy as np
import sys
import codecs
import copy


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import metrics


# 5-fold cross validation.
def five_fold_cross_validation(vec_name, vectorizer, kind, kernel,
     train_labels, train_text, test_labels, test_text, f):
    if kind == 0:
        exp_name = vec_name
    elif kind == 1:
        exp_name = "%s - %s" % (vec_name, kernel)

    f.write(' * ' + exp_name + ':	')
    total_acc = 0.0
    trained_vectorizer = copy.deepcopy(vectorizer)
    train_text_feat = trained_vectorizer.fit_transform(train_text)
    test_text_feat = trained_vectorizer.transform(test_text)

    if kind == 0:
        trained_clf = MultinomialNB().fit(train_text_feat, train_labels)
    elif kind == 1:
        trained_clf = svm.SVC(kernel=kernel).fit(train_text_feat, train_labels)
        trained_clf.fit(train_text_feat, train_labels)

    if kind == 0:
        #print(trained_clf.classes_)
        predicted_prob = trained_clf.predict_proba(test_text_feat)
        #print(predicted_prob)
        #print('\n  -> Accuracy: %.5f\n' % (acc))

    predicted = trained_clf.predict(test_text_feat)
    predicted[0] = 'hi'

    j = 0; correct = 0.0
    for label in test_labels:
        if predicted[j] == label:
            correct = correct + 1
        j = j + 1
    acc = correct / j
    total_acc += acc

    f.write('Accuracy avg: %.3f\n' % ((total_acc) * 100))
    return total_acc

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

# Make vectorizers and kernels.
vectorizers = [
	('freq-1gram', CountVectorizer(ngram_range=(1, 1))),
	('freq-2gram', CountVectorizer(ngram_range=(1, 2))),
	('freq-3gram', CountVectorizer(ngram_range=(1, 3))),
	('tfidf-1gram', TfidfVectorizer(ngram_range=(1, 1))),
	('tfidf-2gram', TfidfVectorizer(ngram_range=(1, 2))),
	('tfidf-3gram', TfidfVectorizer(ngram_range=(1, 3)))
]
kernels = ['linear', 'rbf', 'poly']

f = open('result_svm.txt', 'w')
# SVM
f.write('\n[SVM Classifier]\n')
for vec_name, vectorizer in vectorizers:
	for kernel in kernels:
		five_fold_cross_validation(vec_name, vectorizer, 1, kernel, 
				train_labels, train_text, test_labels, test_text, f)
f.close()
