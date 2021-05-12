import os, random
from tifffile import imread, imsave
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
import wget
import shutil
from enum import Enum
import pandas as pd

from . import models
from . import bcolors

def check_image_dims(image,patch_size):
    # This will open a randomly chosen dataset input image
    x = image
    Image_Y = x.shape[0]
    Image_X = x.shape[1]
    if patch_size > min(Image_Y, Image_X):
        patch_size = min(Image_Y, Image_X)
    print (bcolors.WARNING + " Your chosen patch_size is bigger than the xy dimension of your image; therefore the patch_size chosen is now:",patch_size)
    
    # Here we check that patch_size is divisible by 8
    if not patch_size % 8 == 0:
        patch_size = ((int(patch_size / 8)-1) * 8)
        print (bcolors.WARNING + " Your chosen patch_size is not divisible by 8; therefore the patch_size chosen is now:",patch_size)

    return patch_size

def display_image(image,filename=None):

    # '/content/TrainingDataExample_N2V2D.png'
    norm = simple_norm(image, percent = 99)

    f=plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.imshow(image, interpolation='nearest', norm=norm, cmap='magma')
    plt.title('Training source')
    plt.axis('off')
    if filename != None:
        plt.savefig(filename,bbox_inches='tight',pad_inches=0)
    plt.show()

def check_model_exists(h5_file_path):
    if not os.path.exists(h5_file_path):
        print(bcolors.WARNING+'WARNING: weights_last.h5 pretrained model does not exist')
        return os.path.exists(h5_file_path)
    # If the model path contains a pretrain model, we load the training rate, 


#here we check that no model with the same name already exist, if so print a warning
def check_for_prexisiting_model(model_path,model_name):
    check_model = os.path.exists(model_path+'/'+model_name)
    if check_model:
        print(bcolors.WARNING +"!! WARNING: "+model_name+" already exists and will be deleted in the following cell !!")
        print(bcolors.WARNING +"To continue training "+model_name+", choose a new model_name here, and load "+model_name+" in section 3.3")
        assert not(check_model)
    return check_model

def check_data(image):
    # This will open a randomly chosen dataset input image
    x = image
    len_dims  = len(x.shape)
    if not len_dims == 2:
        print(bcolors.WARNING + "Your images appear to have the wrong dimensions. Image dimension", x.shape)  
        assert len_dims == 2
    # Here we check that the input images contains the expected dimensions
    if len(x.shape) == 2:
        print("Image dimensions (y,x)",x.shape)
    return len_dims

def get_random_image(Training_source):
    random_choice = random.choice(os.listdir(Training_source))
    image = imread(Training_source+"/"+random_choice)
    return image