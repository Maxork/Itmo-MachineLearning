import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_data(filename):
    return pd.read_csv(filename, header=None).values


def split_dataset(test_size):
    dataset = load_data('data_banknote_authentication.txt')
    money_attr = dataset[:, 0:-1]
    money_attr = money_attr.astype(np.float64)
    money_class = dataset[:, -1]
    money_class = money_class.astype(np.float64, copy=False)
    data_train, data_test, class_train, class_test = train_test_split(money_attr, money_class, test_size=test_size,
                                                               				       random_state=55)
    return data_train, class_train, data_test, class_test

def main():
    Experiment(0.1)
    Experiment(0.2)
    Experiment(0.3)
    Experiment(0.4)

def Experiment(testSize):
        data_train, class_train, data_test, class_test = split_dataset(testSize)
        desForest = DecisionTreeClassifier()
        ranForest = RandomForestClassifier()
        ranForest = ranForest.fit(data_train, class_train)
        desForest = desForest.fit(data_train, class_train)
        desisionAccuracy = desForest.score(data_test, class_test)
        randomAccuracy = ranForest.score(data_test, class_test)
        print('Test Size:', testSize)
        print("DecisionTree accuracy: ", desisionAccuracy)
        print("RandomTree accuracy: ", randomAccuracy)
        print('---------------------------------------------')

main()
