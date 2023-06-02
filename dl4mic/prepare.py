# --------------------- Here we delete the model folder if it already exist ------------------------
from . import bcolors
import shutil
import os
import matplotlib.pyplot as plt
from . import reporting
from . import bcolors


def setup_complete(X_train,X_test):

    X = X_train
    validation = X_test

    print("Setup done.")
    # creates a plot and shows one training patch and one validation patch.
    plt.figure(figsize=(16,87))
    plt.subplot(1,2,1)
    plt.imshow(X[0,...,0], cmap='magma')
    plt.axis('off')
    plt.title('Training Patch');
    plt.subplot(1,2,2)
    plt.imshow(validation[0,...,0], cmap='magma')
    plt.axis('off')
    plt.title('Validation Patch');

    # reporting.pdf_export(pretrained_model = Use_pretrained_model)
