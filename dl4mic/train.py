# --------------------- Here we delete the model folder if it already exist ------------------------
from . import bcolors
import shutil
import os
import matplotlib.pyplot as plt
from . import pdf_export
from . import bcolors

def delete_model_if_folder(model_path,model_name):
    if os.path.exists(model_path+'/'+model_name):
        print(bcolors.WARNING +"!! WARNING: Model folder already exists and has been removed !!")
        shutil.rmtree(model_path+'/'+model_name)

def setup_complete(self,X_train,X_test,Use_pretrained_model):

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

    pdf_export(pretrained_model = Use_pretrained_model)