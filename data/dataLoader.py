import pandas as pd
import numpy as np
import _pickle as pkl
import sys
import os
from tqdm import tqdm

NUM_LABEL = 20
NUM_SAMPLE = 3000


def read_data(mode='train'):
    available_mode = ['train', 'test', 'valid']
    if mode not in available_mode:
        raise ValueError("In Reading data: Type Given Error.")
    script_path = sys.path[0]
    data_path = os.path.join(script_path, 'data', 'clean_'+mode+".csv")
    data = np.array(pd.read_csv(data_path))
    return data


def read_features():
    with open(os.path.join(sys.path[0], 'data', 'features.pkl'), 'rb') as file:
        features = pkl.load(file)
    return features


def extract_feature(text, features):
    text = text.split()
    word_bag = set(text)
    feature_list = list([])
    for feature_book in features:
        labeled_features = np.zeros([NUM_SAMPLE])
        for i, feature in enumerate(feature_book):
            if feature in word_bag:
                labeled_features[i] = labeled_features[i] + 1
        feature_list.append(labeled_features)

    return np.stack(feature_list, 0)  # [N_features, N_Sample]


def load_data(mode='train'):
    available_mode = ['train', 'test', 'valid']
    if mode not in available_mode:
        raise ValueError("In Reading data: Type Given Error.")
    data = read_data(mode)
    features = read_features()
    x = list([])
    y = list([])
    print("Loading data...")
    for raw_sample in tqdm(data):
        text = raw_sample[0]
        target = raw_sample[1]
        sample_x = extract_feature(text, features)

        x.append(sample_x)
        y.append(target)

    x = np.stack(x, 0)
    y = np.stack(y, 0)
    print("{} Data loaded with {} samples.".format(mode, x.shape[0]))
    return x, y
