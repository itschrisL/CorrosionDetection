from pathlib import Path

from PIL import Image
from skimage import io
#from Corrosion_Detection_Model import CorrosionDetectionModel
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Helper method that will get all the files in a directory and return it as a list
def load_data_from_folder(img_dir, label_img_dir):
    # try to get the files from the directory
    try:
        img_files = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
        label_img_files = [f for f in listdir(label_img_dir) if isfile(join(label_img_dir, f))]
    except NotADirectoryError:
        print("Could not load files from " + str(img_dir) + " directory")
        return None

    rtrn_data = []
    for i in range(0, len(img_files)):
        rtrn_data.append((Image.open(img_dir/img_files[i]), Image.open(label_img_dir/label_img_files[i])))

    # Shuffle the data here to make it easier in the future
    np.random.shuffle(rtrn_data)
    return rtrn_data


# Helper method that gets the rgb values of an image and returns a 2D array of tuples of rgb values
# This can either take one image or a tuple of two images
def get_rgb_of_img(img):
    if isinstance(img, tuple):
        return (list(img[0].getdata()), list(img[1].getdata()))
    else:
        return list(img.getdata())


def load_data_to_file(images):
    f = open(RGB_DATA_FILE_NAME, 'w')
    for im in images:
        f.write(get_rgb_of_img(im[0]))
        f.write(" || ")
        f.write(get_rgb_of_img(im[1]))
        f.write("\n")
    f.close()

# Helper method to read the data from the file
def read_data_from_file():
    rtn_list = []
    f = open(RGB_DATA_FILE_NAME, 'r')
    lines = f.readlines()
    for line in lines:
        line.replace("\n", "")
        x = line.split(" || ")
        rtn_list.append((eval(x[0]), eval(x[1])))


# Helper method for pre processing the data
def data_pre_processing():
    pass


if __name__ == "__main__":

    RGB_DATA_FILE_NAME = 'rgb_data_file.txt'

    cherryPickedFolderPath = Path("./Images/cherrypicked")
    DATA_FOLDER_PATH = Path()  # Our image file

    data_tuple = load_data_from_folder(Path("./Images/cherrypicked"), Path("./Images/cherrypicked_gt"))
    print("Done loading images.")

    rgb_values = []
    for d in data_tuple:
        rgb_values.append((get_rgb_of_img(d[0]), get_rgb_of_img(d[1])))
    print("Done getting rgb")

