import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def load_data(filename):
    return pd.read_csv(filename, header=None).values


def split_dataset(test_size):
    dataset = load_data('data_banknote_authentication.txt')
    money_attr = dataset[:, 0:-1]
    money_class = dataset[:, -1]
    money_class = money_class.astype(np.int64, copy=False)
    data_train, data_test, class_train, class_test = train_test_split(money_attr, money_class, test_size=test_size,
                                                               				       random_state=55)
    return data_train, class_train, data_test, class_test



def separate_by_class(data_train, class_train):
    classes_dict = {}
    for i in range(len(data_train)):
        classes_dict.setdefault(class_train[i], []).append(data_train[i])
    return classes_dict



def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stand_dev(numbers):  
    var = sum([pow(x - mean(numbers), 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(var)


def summarize(data_train): 
    
    summaries = [(mean(att_numbers), stand_dev(att_numbers)) for att_numbers in zip(*data_train)]
    return summaries



def summarize_by_class(data_train, class_train):
   
    classes_dict = separate_by_class(data_train, class_train)
    summaries = {}
    for class_name, instances in classes_dict.items():
        
        summaries[class_name] = summarize(instances)
    return summaries



def calc_probability(x, mean, stdev):
    if stdev == 0:
        stdev += 0.000001  
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent



def calc_class_probabilities(summaries, instance_attr):
    probabilities = {}
    for class_name, class_summaries in summaries.items():
        probabilities[class_name] = 1.0
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            x = float(instance_attr[i])
            probabilities[class_name] *= calc_probability(x, mean, stdev)
    return probabilities








def predict_one(summaries, instance_attr):
    probabilities = calc_class_probabilities(summaries, instance_attr)
    best_class, max_prob = None, -1
    for class_name, probability in probabilities.items():
        if best_class is None or probability > max_prob:
            max_prob = probability
            best_class = class_name
    return best_class



def predict(summaries, data_test):
    predictions = []
    for i in range(len(data_test)):
        result = predict_one(summaries, data_test[i])
        predictions.append(result)
    return predictions



def calc_accuracy(summaries, data_test, class_test):
    correct_answ = 0
    predictions = predict(summaries, data_test)
    for i in range(len(data_test)):
        if class_test[i] == predictions[i]:
            correct_answ += 1
    return correct_answ / float(len(data_test))
def main():
    data_train, class_train, data_test, class_test = split_dataset(0.3)
    summaries = summarize_by_class(data_train, class_train)
    accuracy = calc_accuracy(summaries, data_test, class_test)
    print('myNBClass ', 'Accuracy: ', accuracy)

    clf = GaussianNB()
    clf.fit(data_train, class_train)
    print('sklNBClass ', 'Accuracy: ', clf.score(data_test, class_test))
    with open('result.NaiveBayes.txt', 'w') as ouf:
        ouf.writelines('myNBClass  ' + 'Accuracy: ' + str(calc_accuracy(summaries, data_test, class_test)) + '\n')
        ouf.writelines('sklNBClass ' + 'Accuracy: '+ str(clf.score(data_test, class_test)))
main()
