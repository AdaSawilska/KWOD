import csv
import os

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom as pydicom
import scipy
from PIL import Image
from natsort import natsorted
import cv2
# Creates new column with two classes 0-normal, 1-pathologic
from scipy.io import loadmat


def labelCreator():
    path = './dataset'
    list_of_images = natsorted(os.listdir(path))
    with open('ROI_labels.csv', 'w', newline='') as csvfile:
        header_key = ['ROI', 'Class']
        new_val = csv.DictWriter(csvfile, fieldnames=header_key)
        new_val.writeheader()
        for img_name in list_of_images:
            new_val.writerow({header_key[0]: img_name, header_key[1]: np.where('norm' in img_name, 0, 1)})



# Przekształca zdjęcia bsifem
def readBreastImages():
    path = './dataset'
    img = []
    list_of_images = natsorted(os.listdir(path))
    for i, image in enumerate(list_of_images):
        with open(path+f'/{image}', 'rb') as pgmf:
            im = pydicom.dcmread(pgmf).pixel_array
            im = filtersBSIF(im)
            img.append(np.asarray(im))
    img = np.asarray(img)
    h5f = h5py.File('ROIbsif11_11_5.h5', 'w')
    h5f.create_dataset('images', data=img)
    h5f.close()
    print('done')

# zapisanie całej macierzy deskryptora do pliku h5
def readDescriptors():
    descriptor = loadmat("./input/HOG11.mat")   # tu wpisujemy ścieżkę do macierzy
    h5f = h5py.File('hog_11.h5', 'w')           # tu wpisujemy nazwę jaką ma mieć plik h5
    h5f.create_dataset('images', data=descriptor["J"])
    h5f.close()
    print("done")


# nałożenie BSIF'u na obraz
def filtersBSIF(image):
    mat1 = scipy.io.loadmat('texturefilters/ICAtextureFilters_11x11_5bit.mat')
    kernel1 = np.ascontiguousarray(mat1['ICAtextureFilters'].T)
    filtered=cv2.filter2D(image, cv2.CV_64F, kernel1[4])
    ret, bin_image = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY)
    return bin_image

if __name__ == '__main__':
    #labelCreator()
    #readBreastImages()
    #savetoCSV()
    readDescriptors()
    #readMSER()



