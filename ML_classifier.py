

import h5py
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from tpot import TPOTClassifier

with h5py.File('./input/all_mias_scans.h5', 'r') as scan_h5:
    bg_info = scan_h5['BG'][:]
    class_info = scan_h5['CLASS'][:]
    #binarized labels - norm or pathology
    data = pd.read_csv("labels.csv")
    class_info = data["7"].values
    # low res scans
    scan_lr = scan_h5['scan'][:][:, ::16, ::16]

scan_lr_flat = scan_lr.reshape((scan_lr.shape[0], -1))

class_le = LabelEncoder()
class_le.fit(class_info)
class_vec = class_le.transform(class_info)
class_le.classes_


idx_vec = np.arange(scan_lr_flat.shape[0])
x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(scan_lr_flat,
                                                                         class_vec,
                                                                         idx_vec,
                                                                         random_state=2017,
                                                                         test_size=0.5,
                                                                         stratify=class_vec)
print('Training', x_train.shape)
print('Testing', x_test.shape)

creport = lambda gt_vec, pred_vec: classification_report(gt_vec, pred_vec, target_names=['NORM', 'PATH'])
                                                         # target_names=[x.decode() for x in
                                                         #               class_le.classes_])
# Most Frequent Model
dc = DummyClassifier(strategy='most_frequent')
dc.fit(x_train, y_train)
y_pred = dc.predict(x_test)
print('Accuracy %2.2f%%' % (100 * accuracy_score(y_test, y_pred)))
print(creport(y_test, y_pred))

# KNearestNeighbors
knn = KNeighborsClassifier(8)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print('Accuracy %2.2f%%' % (100 * accuracy_score(y_test, y_pred)))
print(creport(y_test, y_pred))

# Random Forest
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
print('Accuracy %2.2f%%' % (100 * accuracy_score(y_test, y_pred)))
print(creport(y_test, y_pred))

# Gradient Boosting
xgc = XGBClassifier(silent=False, nthread=2)
xgc.fit(x_train, y_train)
y_pred = xgc.predict(x_test)
print('Accuracy %2.2f%%' % (100 * accuracy_score(y_test, y_pred)))
print(creport(y_test, y_pred))

# AutoML


tpc = TPOTClassifier(generations=2, population_size=5, verbosity=True)
tpc.fit(x_train, y_train)
y_pred = tpc.predict(x_test)
print('Accuracy %2.2f%%' % (100 * accuracy_score(y_test, y_pred)))
print(creport(y_test, y_pred))

print("done")