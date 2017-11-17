from __future__ import division
import pandas as pd
import numpy as np
import operator
from sklearn.model_selection import train_test_split
from math import sqrt
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier



def load_data():
    dataset = pd.read_csv('data_banknote_authentication.txt', header=None).values
    money_attr = dataset[:, 0:-1]
    money_class = dataset[:, -1]
    money_class = money_class.astype(np.int64, copy=False)
    return train_test_split(money_attr, money_class, test_size=0.35)

def euclidean_distance(instance1, instance2):
    squares = [(i - j) ** 2 for i, j in zip(instance1, instance2)]
    return sqrt(sum(squares))



def get_neighbours(instance, data_train, class_train, k):
    distances = []
    for i in data_train:
        distances.append(euclidean_distance(instance, i))
    distances = tuple(zip(distances, class_train))

    return sorted(distances, key=operator.itemgetter(0))[:k]




def get_response(neigbours):
    return Counter(neigbours).most_common()[0][0][1]


def get_predictions(data_train, class_train, data_test, k):
    predictions = []
    for i in data_test:
        neigbours = get_neighbours(i, data_train, class_train, k)
        response = get_response(neigbours)
        predictions.append(response)
    return predictions



def get_accuracy(data_train, class_train, data_test, class_test, k):
    predictions = get_predictions(data_train, class_train, data_test, k)
    mean = [i == j for i, j in zip(class_test, predictions)]
    return sum(mean) / len(mean)

def main():
    data_train, data_test, class_train, class_test = load_data()
    print('myKNClass', 'Accuracy: ', get_accuracy(data_train, class_train, data_test, class_test, 15))

    clf = KNeighborsClassifier(n_neighbors=15)
    clf.fit(data_train, class_train)
    print('sklKNClass', 'Accuracy: ', clf.score(data_test, class_test))
    with open('result.NearestNeighbours.txt', 'w') as ouf:
        ouf.writelines('myKNClass  ' + 'Accuracy: ' + str(get_accuracy(data_train, class_train, data_test, class_test, 15)) + '\n')
        ouf.writelines('sklKNClass ' + 'Accuracy: ' + str(clf.score(data_test, class_test)))

main()

