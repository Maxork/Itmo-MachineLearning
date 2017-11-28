import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn import metrics

def load_data(filename):
    return pd.read_csv(filename, header=None).values


def split_data():
    dataset = load_data('data_banknote_authentication.txt')
    money_attr = dataset[:,-1]
    money_attr = money_attr.astype(np.float64)
    money_class = dataset[:,:-1]
    money_class = money_class.astype(np.float64, copy=False)
    return money_class, money_attr


def data_description(money_attr, money_class):
    columns = ['VarianceOWTI', 'SkewnessOWTI', 'CurtosisOWTI', 'EntropyOWTI', 'Class']
    data = pd.DataFrame(load_data('data_banknote_authentication.txt'), columns=columns)
    print(data.head())

    print('Number of records:', money_class.shape[0])
    print('Number of signs:', money_attr.shape[1])

    print('\nThe shares of each of the classes')
    print('Class 0 (Fake): {:.2%}'.format(list(money_class).count(0) / money_class.shape[0]))
    print('Class 1 (Real): {:.2%}'.format(list(money_class).count(1) / money_class.shape[0]))


def data_2D_visualization(money_attr, money_class):
    plt.figure(figsize=(6, 5))
    for label, marker, color in zip(
            range(0, 2), ('x', 'o'), ('black', 'blue')):
        # Вычисление коэффициента корреляции Пирсона
        R = pearsonr(money_attr[:, 0][money_class == label], money_attr[:, 1][money_class == label])
        plt.scatter(x=money_attr[:, 0][money_class == label],
                    y=money_attr[:, 1][money_class == label],
                    marker=marker,
                    color=color,
                    alpha=0.7,
                    label='class {:}, R={:.2f}'.format(label, R[0])
                    )

    plt.title('Banknote authentication Data Set')
    plt.xlabel('VarianceOWTI')
    plt.ylabel('SkewnessOWTI')
    plt.legend(loc='upper right')
    plt.show()


# def data_3D_visualization(money_attr, money_class):
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     for label, marker, color in zip(
#             range(0, 2), ('x', 'o'), ('red', 'green')):
#         # Вычисление коэффициента корреляции Пирсона
#         ax.scatter(money_attr[:, 0][money_class == label],
#                    money_attr[:, 1][money_class == label],
#                    money_attr[:, 2][money_class == label],
#                    marker=marker,
#                    color=color,
#                    s=40,
#                    alpha=0.7,
#                    label='class {:}'.format(label)
#                    )
#
#     ax.set_xlabel('VarianceOWTI')
#     ax.set_ylabel('SkewnessOWTI')
#     ax.set_zlabel('CurtosisOWTI')
#
#     plt.title('Banknote authentication Data Set')
#     plt.legend(loc='upper right')
#     plt.show()


def train_test_visualization(data_train, data_test, class_train, class_test):
    std_scale = preprocessing.StandardScaler().fit(data_train)
    data_train = std_scale.transform(data_train)
    data_test = std_scale.transform(data_test)
    f, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))

    for a, x_dat, y_lab in zip(ax, (data_train, data_test), (class_train, class_test)):

        for label, marker, color in zip(
                range(0, 2), ('x', 'o'), ('black', 'blue')):
            a.scatter(x=x_dat[:, 0][y_lab == label],
                      y=x_dat[:, 1][y_lab == label],
                      marker=marker,
                      color=color,
                      alpha=0.7,
                      label='class {}'.format(label)
                      )

        a.legend(loc='upper right')

    ax[0].set_title('VarianceOWTI')
    ax[1].set_title('SkewnessOWTI')
    f.text(0.5, 0.04, 'VarianceOWTI (standardized)', ha='center', va='center')
    f.text(0.08, 0.5, 'SkewnessOWTI (standardized)', ha='center', va='center', rotation='vertical')

    # plt.show()


def train_data(money_attr, money_class):
    data_train, data_test, class_train, class_test = train_test_split(money_attr, money_class, test_size=0.30,
                                                                      random_state=123)
    print('\nThe shares of each of the classes')
    print('\nTraining Dataset:')
    print('Class 0 (Fake): {:.2%}'.format(list(class_train).count(0) / class_train.shape[0]))
    print('Class 1 (Real): {:.2%}'.format(list(class_train).count(1) / class_train.shape[0]))

    print('\nTest Dataset:')
    print('Class 0 (Fake): {:.2%}'.format(list(class_test).count(0) / class_test.shape[0]))
    print('Class 1 (Real): {:.2%}'.format(list(class_test).count(1) / class_test.shape[0]))

    train_test_visualization(data_train, data_test, class_train, class_test)
    return data_train, data_test, class_train, class_test


def linear_discriminant_analysis(data_train, class_train):
    sklearn_lda = LDA()
    sklearn_transf = sklearn_lda.fit(data_train, class_train).transform(data_train)

    plt.figure(figsize=(8, 8))
    for label, marker, color in zip(
            range(0, 2), ('x', 'o'), ('black', 'blue')):
        plt.scatter(x=sklearn_transf[class_train == label],
                    y=sklearn_transf[class_train == label],
                    marker=marker,
                    color=color,
                    alpha=0.7,
                    label='class {}'.format(label))

    plt.xlabel('vector 1')
    plt.ylabel('vector 2')

    plt.legend()
    # Визуализация разбиения классов после линейного преобразования LDA
    plt.title('Most significant singular vectors after linear transformation via LDA')

    plt.show()


def train_linear_discriminant_analysis(data_train, data_test, class_train, class_test):
    lda_clf = LDA()
    lda_clf.fit(data_train, class_train)
    # LDA(n_components=None, priors=None, shrinkage=None, solver='svd', store_covariance=False, tol=0.0001)

    pred_train = lda_clf.predict(data_train)
    print('LDA')
    print('Точность классификации на тестовом наборе данных')
    print('{:.2%}'.format(metrics.accuracy_score(class_train, pred_train)))

    pred_test = lda_clf.predict(data_test)

    print('Точность классификации на обучающем наборе данных')
    print('{:.2%}'.format(metrics.accuracy_score(class_test, pred_test)))


def train_quadratic_discriminant_analysis(data_train, data_test, class_train, class_test):
    qda_clf = QDA()
    qda_clf.fit(data_train, class_train)

    pred_train = qda_clf.predict(data_train)
    print('__________________________________________________')
    print('QDA')
    print('Точность классификации на тестовом наборе данных')
    print('{:.2%}'.format(metrics.accuracy_score(class_train, pred_train)))

    pred_test = qda_clf.predict(data_test)

    print('Точность классификации на обучающем наборе данных')
    print('{:.2%}'.format(metrics.accuracy_score(class_test, pred_test)))


def main():
    money_attr, money_class = split_data()

    data_description(money_attr, money_class)
    data_2D_visualization(money_attr, money_class)
    #data_3D_visualization(money_attr, money_class)

    data_train, data_test, class_train, class_test = train_data(money_attr, money_class)

    linear_discriminant_analysis(data_train, class_train)
    train_linear_discriminant_analysis(data_train, data_test, class_train, class_test)
    train_quadratic_discriminant_analysis(data_train, data_test, class_train, class_test)


main()


