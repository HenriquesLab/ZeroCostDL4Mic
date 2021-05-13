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

