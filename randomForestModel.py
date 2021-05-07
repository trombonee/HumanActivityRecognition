import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from datetime import datetime
warnings.simplefilter("ignore")

x_train = np.load('trainingFeatureData.npy').astype(float)
x_test = np.load('testFeatureData.npy').astype(float)
y_test = np.load('y_test.npy').astype(int)
y_train = np.load('y_train.npy').astype(int)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

table = pd.DataFrame(columns = ["Model", "Accuracy(%)"])
def keeping_record(model_name, accuracy):
    global table
    table = table.append(pd.DataFrame([[model_name, accuracy]], columns = ["Model", "Accuracy(%)"]))
    table.reset_index(drop = True, inplace = True)

def print_confusionMatrix(Y_TestLabels, PredictedLabels):
    confusionMatx = confusion_matrix(Y_TestLabels, PredictedLabels)
    
    precision = confusionMatx/confusionMatx.sum(axis = 0)
    
    recall = (confusionMatx.T/confusionMatx.sum(axis = 1)).T
    
    sns.set(font_scale=1.5)
    
    # confusionMatx = [[1, 2],
    #                  [3, 4]]
    # confusionMatx.T = [[1, 3],
    #                   [2, 4]]
    # confusionMatx.sum(axis = 1)  axis=0 corresponds to columns and axis=1 corresponds to rows in two diamensional array
    # confusionMatx.sum(axix =1) = [[3, 7]]
    # (confusionMatx.T)/(confusionMatx.sum(axis=1)) = [[1/3, 3/7]
    #                                                  [2/3, 4/7]]

    # (confusionMatx.T)/(confusionMatx.sum(axis=1)).T = [[1/3, 2/3]
    #                                                    [3/7, 4/7]]
    # sum of row elements = 1
    
    labels = ["SHOOTING", "DRIBBLING", "JUMPING JACKS", "WALKING", "BASEBALL SWING" ]

    plt.figure(figsize=(16, 11.5))
    sns.heatmap(confusionMatx, cmap="Blues", annot=True, fmt=".1f", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix", fontsize=30)
    plt.xlabel('Predicted Class', fontsize=20)
    plt.ylabel('Original Class', fontsize=20)
    plt.tick_params(labelsize=15)
    plt.xticks(rotation=30)
    plt.yticks(rotation=0)
    plt.show()

    print("-" * 125)

    plt.figure(figsize=(16, 11.5))
    sns.heatmap(precision, cmap="Blues", annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels)
    plt.title("Precision Matrix", fontsize=30)
    plt.xlabel('Predicted Class', fontsize=20)
    plt.ylabel('Original Class', fontsize=20)
    plt.tick_params(labelsize=15)
    plt.xticks(rotation=30)
    plt.yticks(rotation=0)
    plt.show()

    print("-" * 125)

    plt.figure(figsize=(16, 11.5))
    sns.heatmap(recall, cmap="Blues", annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, vmax=1)
    plt.title("Recall Matrix", fontsize=30)
    plt.xlabel('Predicted Class', fontsize=20)
    plt.ylabel('Original Class', fontsize=20)
    plt.tick_params(labelsize=15)
    plt.xticks(rotation=30)
    plt.yticks(rotation=0)
    plt.show()

def apply_model(cross_val, x_train, y_train, x_test, y_test, model_name):
    start = datetime.now()
    cross_val.fit(x_train, y_train)
    predicted_points = cross_val.predict(x_test)
    
    print("Total time taken for tuning hyperparameter and making prediction by the model is (HH:MM:SS): {}\n".format(datetime.now() - start))
    accuracy = np.round(accuracy_score(y_test, predicted_points)*100, 2)
    
    print('---------------------')
    print('|      Accuracy      |')
    print('---------------------')
    print(str(accuracy)+"%\n")
    
    print('---------------------------')
    print('|      Best Estimator      |')
    print('---------------------------')
    print("{}\n".format(cross_val.best_estimator_))
    
    print('----------------------------------')
    print('|      Best Hyper-Parameters      |')
    print('----------------------------------')
    print(cross_val.best_params_)
    
    keeping_record(model_name, accuracy)
    
    print("\n\n")
    
    print(y_test, predicted_points)
    predicted_points = [np.argmax(i)+1 for i in predicted_points]
    y_labels = [np.argmax(i)+1 for i in y_test]
    print_confusionMatrix(y_labels, predicted_points)

parameters = {"n_estimators": [50, 100, 200, 400, 800]}
clf = RandomForestClassifier()
cross_val = GridSearchCV(clf, parameters, cv=3)
apply_model(cross_val, x_train, y_train, x_test, y_test, "Random Forest")