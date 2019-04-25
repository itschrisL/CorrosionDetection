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
        rtrn_data.append((img_dir/img_files[i], label_img_dir/label_img_files[i]))
    return rtrn_data

def data_pre_processing():
    pass


if __name__ == "__main__":
    cherryPickedFolderPath = Path("./Images/cherrypicked")
    DATA_FOLDER_PATH = Path()  # Our image file
    img = Image.open("./Images/cherrypicked/image004.jpg")
    WIDTH, HEIGHT = img.size
    print("Width: " + str(WIDTH) + " | Height: " + str(HEIGHT))
    # model = CorrosionDetectionModel()
    data_tuple = load_data_from_folder(Path("./Images/cherrypicked"), Path("./Images/cherrypicked_gt"))

    # for d in data_tuple:
    #     img = Image.open(str(d[0]))
    #     width, height = img.size
    #     print("Width: " + str(width) + " | Height: " + str(height))
