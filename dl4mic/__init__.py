# import tensorflow as tf
# ------- Common variable to all ZeroCostDL4Mic notebooks -------

import numpy as np
from matplotlib import pyplot as plt
import urllib
import os, random
import shutil
import zipfile
from tifffile import imread, imsave
import time
import sys
import wget
from pathlib import Path
import pandas as pd
import csv
from glob import glob
from scipy import signal
from scipy import ndimage
from skimage import io
from sklearn.linear_model import LinearRegression
from skimage.util import img_as_uint
import matplotlib as mpl
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio as psnr
from astropy.visualization import simple_norm
from skimage import img_as_float32
from fpdf import FPDF, HTMLMixin
from datetime import datetime
from pip._internal.operations.freeze import freeze
import subprocess
from datetime import datetime



# def test_tf_gpu():
#     if tf.test.gpu_device_name() == "":
#         print("You do not have GPU access.")
#         print("Did you change your runtime ?")
#         print(
#             "If the runtime setting is correct then Google did not allocate a GPU for your session"
#         )
#         print("Expect slow performance. To access GPU try reconnecting later")
#     else:
#         print("You have GPU access")
#         # !nvidia-smi


class bcolors:
    WARNING = "\033[31m"


W = "\033[0m"  # white (normal)
R = "\033[31m"  # red

ref_1 = 'References:\n - ZeroCostDL4Mic: von Chamier, Lucas & Laine, Romain, et al. "ZeroCostDL4Mic: an open platform to simplify access and use of Deep-Learning in Microscopy." BioRxiv (2020).'

# def __main__():
#     read_latest_notebook_version()


# if (Use_Default_Advanced_Parameters): 
#   print("Default advanced parameters enabled")
#   # number_of_steps is defined in the following cell in this case
#   batch_size = 128
#   percentage_validation = 10
#   initial_learning_rate = 0.0004



