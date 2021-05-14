import matplotlib as plt
from tifffile.tifffile import imread
from . import models
import random
import os


def full(Data_folder, Result_folder, Data_type):
    if not ((Data_folder is None) or (Result_folder is None) or (Data_type is None)):
        display_random_image(Data_folder, Result_folder, Data_type)
    pass


def display_random_image(Data_folder, Result_folder, Data_type):

    random_choice = random.choice(os.listdir(Data_folder))
    x = imread(os.path.join(Data_folder,random_choice))

    os.chdir(Result_folder)
    y = imread(os.path.join(Result_folder,random_choice))

    if Data_type == models.params.Data_type.SINGLE_IMAGES:

        f = plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(x, interpolation="nearest")
        plt.title("Input")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(y, interpolation="nearest")
        plt.title("Predicted output")
        plt.axis("off")
        plt.show()

    if Data_type == models.params.Data_type.STACKS:

        f = plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(x[1], interpolation="nearest")
        plt.title("Input")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(y[1], interpolation="nearest")
        plt.title("Predicted output")
        plt.axis("off")
        plt.show()
