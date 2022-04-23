import csv
import os

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from natsort import natsorted

# Creates new column with two classes 0-normal, 1-pathologic
def newClass():
    data = pd.read_csv('./labels.txt', sep=" ", header=None)
    data.columns = ["name", "1", "2", "3", "4", "5", "6"]
    data["7"] = np.where(data['2'] != 'NORM', 1, 0)

    data.to_csv('labels.csv', index=False)
    print("done")

# Reads photos and save them as array in .h5 file
def readBreastImages():
    path = './input/output_stretch'
    img = []
    list_of_images = natsorted(os.listdir(path))
    for i, image in enumerate(list_of_images):
        with open(path+f'/{image}', 'rb') as pgmf:
            im = Image.open(pgmf)
            #img.append(np.asarray(im.resize((256, 256))))
            img.append(np.asarray(im))
        #plt.imshow(img[i])
        #plt.show()
    img = np.asarray(img)
    h5f = h5py.File('images_stretch.h5', 'w')
    h5f.create_dataset('images', data=img)
    h5f.close()
    print('done')

def savetoCSV():
    results = {"Basia": 10, "Ada": 18, "Jola": 23, "Kasia": 3}

    # save to empty file
    # with open('try.csv', 'w', newline='') as csvfile:
    #     header_key = ['Imie', 'Wiek']
    #     new_val = csv.DictWriter(csvfile, fieldnames=header_key)
    #     new_val.writeheader()
    #     for new_k in results:
    #         new_val.writerow({header_key[0]: new_k, header_key[1]: results[new_k]})

    # add column to existing csv file
    df = pd.read_csv('try.csv')
    df["Waga"]=[34, 56, 63, 10]

    print("read")



if __name__ == '__main__':
    #newClass()
    readBreastImages()
    #savetoCSV()



