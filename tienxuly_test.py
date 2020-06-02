from pyvi import ViTokenizer, ViPosTagger 
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
import gensim
import os 
import io
dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
dir_path = os.path.join(dir_path, 'data')

def get_data(folder_path):
    X = []
    y = []
    dirs = os.listdir(folder_path)
    for path in tqdm(dirs):
        file_paths = os.listdir(os.path.join(folder_path, path))
        for file_path in tqdm(file_paths):
            with io.open(os.path.join(folder_path, path, file_path), 'r', encoding="utf-16") as f:
                lines = f.readlines()
                lines = ' '.join(lines)
                lines = gensim.utils.simple_preprocess(lines)
                lines = ' '.join(lines)
                lines = ViTokenizer.tokenize(lines)
                X.append(lines)
                y.append(path)
                

    return X, y

X_data, y_data = get_data('data')

encoder = preprocessing.LabelEncoder()
y_data_n = encoder.fit_transform(y_data)

encoder.classes_