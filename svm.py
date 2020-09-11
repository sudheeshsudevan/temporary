input_folder = "output"

import cv2
import numpy as np
import os
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt

def list_folder(path):
    # returns a list of names (with extension, without full path) of all files 
    # in folder path
    files = []
    for name in os.listdir(path):
        if os.path.isdir(os.path.join(path, name)):
            files.append(name)
    return files

def list_files(path):
    # returns a list of names (with extension, without full path) of all files 
    # in folder path
    files = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            files.append(name)
    return files

def get_relevant_folder_names_and_labels(path):
    folders, labels = [],[]
    all_folders = list_folder(input_folder)
    for folder in all_folders:
        if folder[4] == 'F' or folder[4] == "O":
            folders.append(folder)
            labels.append(folder[4])
    return folders, labels

def get_image_vectors(folders):
    image_vectors = []
    for folder in folders:
        folder_path = os.path.join(input_folder, folder)
        image_path = os.path.join(folder_path, "agglomerative.png")
        image = cv2.imread(image_path)
        arr = np.array(image).ravel()
        image_vectors.append(arr)
    return np.array(image_vectors)

def svc_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3,random_state=109)
    param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]
    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid)
    clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # print("Classification report for - \n{}:\n{}\n".format(
    # clf, metrics.classification_report(y_test, y_pred)))
    # print("Confusion Matrix:")
    # print(metrics.confusion_matrix(y_test, y_pred))
    metrics.plot_confusion_matrix(clf, X_test, y_test)
    plt.show()

if __name__ == "__main__":
    folders, y = get_relevant_folder_names_and_labels(input_folder)
    X = get_image_vectors(folders)
    svc_classifier(X, y)
