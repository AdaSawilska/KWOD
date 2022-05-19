from os.path import exists

import h5py
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from tpot import TPOTClassifier

filenames = ['hog_2.h5']
for name in filenames:

    h5f = h5py.File(name, 'r')
    # jeśli klasyfikujemy wyjscie deskryptorów np. hog
    descriptors = h5f['images'][:][:]
    scan_lr_flat = descriptors.reshape((descriptors.shape[0], -1))
    idx_vec = np.arange(descriptors.shape[0])

    # jeśli klasyfikujemy całe obrazy np. bsif
    # img = h5f['images'][:][:, :, :]
    # plt.imshow(img[8])
    # plt.show()
    #scan_lr_flat = img.reshape((img.shape[0], -1))
    #idx_vec = np.arange(scan_lr_flat.shape[0])
    h5f.close()

    labels = pd.read_csv("ROI_labels.csv")
    class_info = labels["Class"].values
    class_le = LabelEncoder()
    class_le.fit(class_info)
    class_vec = class_le.transform(class_info)

    # podział danych na testowe i treningowe: jeśli wkładamy deskryptor, to jako pierwszy argument wpisać "descriptors"
    # jeśli obrazy, to wpisać "scan_lr_flat"
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(descriptors, #scan_lr_flat,
                                                                             class_vec,
                                                                             idx_vec,
                                                                             random_state=2017,
                                                                             test_size=0.5,
                                                                             stratify=class_vec)
    print('Training', x_train.shape)
    print('Testing', x_test.shape)

    creport = lambda gt_vec, pred_vec: classification_report(gt_vec, pred_vec, target_names=["0", "1"])

    #Most Frequent Model
    dc = DummyClassifier(strategy='most_frequent')
    dc.fit(x_train, y_train)
    y_pred = dc.predict(x_test)
    print('Accuracy %2.2f%%' % (100 * accuracy_score(y_test, y_pred)))
    results = {"DummyClassifier": round(100 * accuracy_score(y_test, y_pred),2)}
    print(creport(y_test, y_pred))
    #
    # # KNearestNeighbors
    # knn = KNeighborsClassifier(8)
    # knn.fit(x_train, y_train)
    # y_pred = knn.predict(x_test)
    # print('Accuracy %2.2f%%' % (100 * accuracy_score(y_test, y_pred)))
    # results["kNN"] = round(100 * accuracy_score(y_test, y_pred),2)
    # print(creport(y_test, y_pred))
    #
    # # Random Forest
    # rfc = RandomForestClassifier()
    # rfc.fit(x_train, y_train)
    # y_pred = rfc.predict(x_test)
    # print('Accuracy %2.2f%%' % (100 * accuracy_score(y_test, y_pred)))
    # results["RandomForest"] = round(100 * accuracy_score(y_test, y_pred),2)
    # print(creport(y_test, y_pred))

    # Gradient Boosting
    # xgc = XGBClassifier(silent=False, nthread=2)
    # xgc.fit(x_train, y_train)
    # y_pred = xgc.predict(x_test)
    # print('Accuracy %2.2f%%' % (100 * accuracy_score(y_test, y_pred)))
    # results["XGB"] = round(100 * accuracy_score(y_test, y_pred),2)
    # print(creport(y_test, y_pred))

    # SVM
    svm = svm.SVC(kernel='linear') #kernel='rbf'
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    print('Accuracy %2.2f%%' % (100 * accuracy_score(y_test, y_pred)))
    results["SVM"] = round(100 * accuracy_score(y_test, y_pred), 2)
    print(creport(y_test, y_pred))

    # AutoML
    # tpc = TPOTClassifier(generations=2, population_size=5, verbosity=True)
    # tpc.fit(x_train, y_train)
    # y_pred = tpc.predict(x_test)
    # print('Accuracy %2.2f%%' % (100 * accuracy_score(y_test, y_pred)))
    # #results["TPOT"] = round(100 * accuracy_score(y_test, y_pred),2)
    # print(creport(y_test, y_pred))

    # zapis wyników do pliku CSV
    path_to_results = 'results_descriptors.csv'
    if exists(path_to_results)==False:
        with open(path_to_results, 'w', newline='') as csvfile:
            header_key = ['Classifier', 'Accuracy for HOG']
            new_val = csv.DictWriter(csvfile, fieldnames=header_key)
            new_val.writeheader()
            for new_k in results:
                new_val.writerow({header_key[0]: new_k, header_key[1]: results[new_k]})
    else:
        s = name.split(".")
        df = pd.read_csv(path_to_results)
        # jeśli któregoś z klasyfikatorów nie używamy to trzeba wywalić i dać np. jakiś string albo 0
        df[f"Accuracy {s[0]}"] = [results["DummyClassifier"], results["kNN"], results["RandomForest"], results["XGB"], "nic" ]
        df.to_csv(path_to_results, index=False)


    print("done")

