import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm
from sklearn.datasets import make_classification

with open('Heart.csv', 'r') as input_file:
    make_classification = input_file.readlines()
    for index, line in enumerate(make_classification):
          make_classification[index] = line.strip()
    le = preprocessing.LabelEncoder()
    data = pd.DataFrame([sub.split(",") for sub in make_classification], columns=['','Age','sex','ChestPain','RestBP','Chol','Fbs','RestECG','MaxHR','ExAng','Oldpeak','Slope','Ca','Thal','AHD'])
    data1 = data.apply(le.fit_transform)
    X=data1[['','Age','sex','ChestPain','RestBP','Chol','Fbs','RestECG','MaxHR','ExAng','Oldpeak','Slope','Ca','Thal']]
    y=data1[['AHD']]
    X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(y), test_size=0.3)
    C = 1.0
    svc = svm.SVC(kernel='linear', C=C).fit(X, np.ravel(y))
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, np.ravel(y))
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, np.ravel(y))
    svc.fit(X_train, np.ravel(y_train))
    
    #Predicting the response for Linear SVC: training dataset
    y_training_pred = svc.predict(X_train)
    print("Linear SVC: Training set: Accuracy:",metrics.accuracy_score(y_train, y_training_pred))
    
    #Predicting the response for Linear SVC: test dataset
    y_test_linear_pred = svc.predict(X_test)
    print("Linear SVC: Test set: Accuracy:",metrics.accuracy_score(y_test, y_test_linear_pred))
    
     #Predicting the response for Polynomial SVC: test dataset
    y_test_poly_pred = rbf_svc.predict(X_test)
    print("Polynomial SVC: Test set: Accuracy:",metrics.accuracy_score(y_test, y_test_poly_pred))
    
    #Predicting the response for RBF SVC: test dataset
    y_test_rbf_pred = rbf_svc.predict(X_test)
    print("RBF SVC: Test set: Accuracy:",metrics.accuracy_score(y_test, y_test_rbf_pred))
    
    C_2d_range = [1e-2, 1, 1e2]
    trainingSetClassifiers = []
    testingSetclassifiers= []
    for C in C_2d_range:
      clf = svm.SVC(C=C)
      clf.fit(X_train, y_train)
      test_accuracy = metrics.accuracy_score(y_test, clf.predict(X_test))
      training_accuracy = metrics.accuracy_score(y_train, clf.predict(X_train))
      testingSetclassifiers.append((C, test_accuracy))
      trainingSetClassifiers.append((C, training_accuracy))
    x_test_accuracy, y_test_accuracy = zip(*testingSetclassifiers)
    x_train_accuracy, y_train_accuracy = zip(*trainingSetClassifiers)
    plt.xlabel("C value")
    plt.ylabel("Accuracy")
    plt.plot(x_train_accuracy, y_train_accuracy, label="Training set accuracy")
    plt.plot(x_test_accuracy, y_test_accuracy, label="Testing set accuracy")
    plt.legend()
    plt.show()