from pathlib import Path
import tensorflow as tf
from PIL import Image
from scipy._lib.six import xrange
from skimage import io
from Corrosion_Detection_Model import CorrosionDetectionModel
from os import listdir
from os.path import isfile, join
import PIL
from PIL import Image
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
def get_rgb_of_img(img):
    pixels = list(img.getdata())
    width, height = img.size
    pixels = [pixels[i * width:(i + 1) * width] for i in xrange(height)]
    return pixels


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


def show_image(img):
    plt.imshow(img)
    plt.show()


# Helper method that resize an image
def reformat_image(img, new_width=256, new_height=256):
    new_img = img.resize((new_width, new_height))
    return new_img


def find_biggest_resolution(img_list):
    width_max = 0
    height_max = 0
    for im in img_list:
        w, h = im.size
        if w > width_max:
            width_max = w
        if h > height_max:
            height_max = h
    return width_max, height_max


if __name__ == "__main__":

    RGB_DATA_FILE_NAME = 'rgb_data_file.txt'

    cherryPickedFolderPath = Path("./Images/cherrypicked")
    DATA_FOLDER_PATH = Path()  # Our image file

    image_list = load_data_from_folder(Path("./Images/cherrypicked"), Path("./Images/cherrypicked_gt"))
    print("Done loading images.")

    temp_list = []
    for i in image_list:
        temp_list.append(i[0])

    width, height = find_biggest_resolution(temp_list)
    print("Width: " + str(width) + " | Height: " + str(height))

    rgb_values = []
    for d in image_list:
        rgb_values.append((
            get_rgb_of_img(reformat_image(d[0])),
            get_rgb_of_img(reformat_image(d[1]))
        ))

    testing_set = []
    label_set = []
    for i in rgb_values:
        testing_set.append(i[0])
        label_set.append(i[1])

    corrosion_model = CorrosionDetectionModel()
    corrosion_model.train_model(np.array(testing_set), np.array(label_set))





