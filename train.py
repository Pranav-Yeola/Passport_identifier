import numpy as np
import csv
import sys
import pickle
import collections

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

class Node:
    def __init__(self, predicted_class, depth):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.depth = depth
        self.left = None
        self.right = None


def import_data():
    X = np.genfromtxt("train_X_de.csv",delimiter=',',dtype = np.float128,skip_header = 1)
    Y = np.genfromtxt("train_Y_de.csv",delimiter=',',dtype = np.float128)
    return X, Y

def calculate_gini_leaf(Y):
    class_dict = dict(collections.Counter(Y))
    P = np.array(list(class_dict.values()))/len(Y)
    gini = 1 - np.sum(np.square(P))
    return gini

def calculate_gini_index(Y_subsets):
    gini_leafs = np.array([calculate_gini_leaf(subset) for subset in Y_subsets])
    subsets_length = [len(subset) for subset in Y_subsets]
    total_observantions = sum(subsets_length)
    PB = np.array([sub_len/total_observantions for sub_len in subsets_length ])
    total_gini = np.dot(PB,gini_leafs.T)
    return total_gini

def split_data_set(data_X, data_Y, feature_index, threshold):
    X_Y = np.hstack([np.array(data_X),np.array([data_Y]).T])
    Left_XY = X_Y[X_Y[:,feature_index]<threshold,:]
    Right_XY = X_Y[X_Y[:,feature_index]>=threshold,:]
    Left_X = np.array([])
    Left_Y = np.array([])
    Right_X = np.array([])
    Right_Y = np.array([])
    if len(Left_XY)>0:
        Left_X = Left_XY[:,0:len(Left_XY[0])-1]
        Left_Y = Left_XY[:,len(Left_XY[0])-1]
    if len(Right_XY)>0:
        Right_X = Right_XY[:,0:len(Right_XY[0])-1]
        Right_Y = Right_XY[:,len(Right_XY[0])-1]
    
    return Left_X, Left_Y.astype(int), Right_X, Right_Y.astype(int)

def get_best_split(X, Y):
    X= np.array(list(X))
    best_feature = 0
    best_threshold = 0
    best_gini = 9999
    for feature_index in range(len(X[0])):
        thresholds = set(sorted(X[:, feature_index]))
        for threshold in thresholds:
            left_X,left_Y,right_X,right_Y = split_data_set(X, Y, feature_index, threshold)
            if len(left_X) == 0 and len(right_X) ==0:
                continue
            if calculate_gini_index([left_Y,right_Y]) < best_gini:
                best_feature = feature_index
                best_threshold = threshold
                best_gini = calculate_gini_index([left_Y,right_Y])
    return best_feature,best_threshold


def construct_tree(X, Y, max_depth, min_size, depth):
    classes = list(set(Y))
    predicted_class = classes[np.argmax([np.sum(Y == c) for c in classes])]
    node = Node(predicted_class, depth)
    if len(set(Y)) == 1:
        return node
    if depth >= max_depth:
        return node
    if len(Y) <= min_size:
        return node
    
    feature_index, threshold = get_best_split(X, Y)
    if feature_index is None or threshold is None:
        return node

    node.feature_index = feature_index
    node.threshold = threshold
    left_X, left_Y, right_X, right_Y = split_data_set(X, Y, feature_index, threshold)
    node.left = construct_tree(np.array(left_X), np.array(left_Y), max_depth, min_size, depth + 1)
    node.right = construct_tree(np.array(right_X), np.array(right_Y), max_depth, min_size, depth + 1)
    return node


def train_model(X,Y):
    maxdepth = len(X[0])*100
    mindepth = 2
    tree_root = construct_tree(X,Y,maxdepth, mindepth, 0)
    return tree_root


def get_model():
    X,Y = import_data()
    model = train_model(X,Y)
    return model

if __name__ == "__main__":
    X,Y = import_data()
    
    model = train_model(X,Y)
    

