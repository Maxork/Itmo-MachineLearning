from sklearn.neighbors import KNeighborsClassifier 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import numpy as np
from sklearn.model_selection import cross_val_score
#from sklearn import cross_validation
import pandas as pd

dataset = pd.read_csv('data_banknote_authentication.txt', header=None).values
money_attr = dataset[:, :4]
money_attr = money_attr.astype(np.float64)
money_class = dataset[:, -1]
money_class = money_class.astype(np.float64, copy=False)

kFold=sklearn.cross_validation.KFold(n=len(money_attr),n_folds=5, random_state=8)




#metric Accuracy for GaussianNB and LDA
clf = KNeighborsClassifier()
scores = cross_val_score(clf, money_attr, money_class,  cv=kFold, scoring='accuracy')
print("Accuracy for KNeighbors: %0.3f (%0.3f)" % (scores.mean(), scores.std() ))

qda=QDA()
scores = cross_val_score(qda, money_attr, money_class, cv=kFold, scoring='accuracy')
print("Accuracy for QDA: %0.3f (%0.3f)" % (scores.mean(), scores.std() ))

from sklearn.preprocessing import label_binarize

        
scores = cross_val_score(clf,  money_attr, money_class, cv=kFold, scoring='neg_log_loss')
print("log_loss for KNeighbors: %0.3f (%0.3f)" % (scores.mean(), scores.std() ))

scores = cross_val_score(qda, money_attr,money_class,  cv=kFold, scoring='neg_log_loss')
print("log_loss for QDA: %0.3f (%0.3f)" % (scores.mean(), scores.std() ))