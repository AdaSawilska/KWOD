import csv
import os

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom as pydicom
from PIL import Image
from natsort import natsorted

# Creates new column with two classes 0-normal, 1-pathologic
from scipy.io import loadmat


def newClass():
    path = './dataset'
    list_of_images = natsorted(os.listdir(path))
    choices = ["norm", "arch", "spicule"]

    with open('ROI_labels.csv', 'w', newline='') as csvfile:
        header_key = ['ROI', 'Class']
        new_val = csv.DictWriter(csvfile, fieldnames=header_key)
        new_val.writeheader()
        for img_name in list_of_images:
            conditions = ['norm' in img_name, 'arch' in img_name, 'spicule' in img_name]
            new_val.writerow({header_key[0]: img_name, header_key[1]: np.select(conditions, choices, default=np.nan)})


    # data = pd.read_csv('./ROI_labels.csv', sep=" ", header=)
    # data.columns = ["ROI"]
    # data["Class"] = np.where('norm' in data["ROI"])
    #
    # data.to_csv('labels.csv', index=False)

    # with open('ROI_labels.csv', 'w', newline='') as csvfile:
    #     header_key = ['ROI', 'Class']
    #     new_val = csv.DictWriter(csvfile, fieldnames=header_key)
    #     new_val.writeheader()
    #     for new_k in results:
    #         new_val.writerow({header_key[0]: new_k, header_key[1]: results[new_k]})





#
# Reads photos and save them as array in .h5 file
def readBreastImages():
    path = './output_strech'
    img = []
    list_of_images = natsorted(os.listdir(path))
    for i, image in enumerate(list_of_images):
        with open(path+f'/{image}', 'rb') as pgmf:
            #im = Image.open(pgmf)
            im = pydicom.dcmread(pgmf).pixel_array
            # plt.imshow(im, cmap=plt.cm.bone)
            # plt.show()
            img.append(np.asarray(im))
        #plt.imshow(img[i])
        #plt.show()
    img = np.asarray(img)
    h5f = h5py.File('ROIstrech.h5', 'w')
    h5f.create_dataset('images', data=img)
    h5f.close()
    print('done')
#
# def savetoCSV():
#     results = {"Basia": 10, "Ada": 18, "Jola": 23, "Kasia": 3}
#
#     # save to empty file
#     # with open('try.csv', 'w', newline='') as csvfile:
#     #     header_key = ['Imie', 'Wiek']
#     #     new_val = csv.DictWriter(csvfile, fieldnames=header_key)
#     #     new_val.writeheader()
#     #     for new_k in results:
#     #         new_val.writerow({header_key[0]: new_k, header_key[1]: results[new_k]})
#
#     # add column to existing csv file
#     df = pd.read_csv('try.csv')
#     df["Waga"]=[34, 56, 63, 10]
#
#     print("read")
#
def readDescriptors():
    descriptor = loadmat("./input/HOG_wyniki.mat")
    h5f = h5py.File('hog_original.h5', 'w')
    h5f.create_dataset('images', data=descriptor["J"])
    h5f.close()
    print("done")




if __name__ == '__main__':
    #newClass()
    #readBreastImages()
    #savetoCSV()
    readDescriptors()



