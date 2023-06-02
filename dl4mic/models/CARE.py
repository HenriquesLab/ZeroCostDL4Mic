# from __future__ import print_function, unicode_literals, absolute_import, division

import os
from random import triangular
import shutil
from dl4mic.reporting import pdf_export
import time
import numpy as np
import csv
import pandas as pd
from .. import models



# ------- Variable specific to CARE -------
from csbdeep.utils import (
    download_and_extract_zip_file,
    plot_some,
    axes_dict,
    plot_history,
    Path,
    download_and_extract_zip_file,
)
from csbdeep.data import RawData, create_patches
from csbdeep.io import load_training_data, save_tiff_imagej_compatible
# from csbdeep.models import Config, CARE
from csbdeep import data
import csbdeep.models

from typing import List

# def __init__(self):
#     return self.N2V

# from models import params

# default_params = {
#     "model": "CARE",
#     "model_name": None,
#     "model_path": None,
#     "ref_str": None,
#     "Notebook_version": 1.12,
#     "initial_learning_rate": 0.0004,
#     "number_of_steps": 400,
#     "number_of_patches": 100,
#     "percentage_validation": 10,
#     "image_patches": None,
#     "loss_function": None,
#     "batch_size": 16,
#     "patch_size": 80,
#     "Training_source": None,
#     "number_of_epochs": 100,
#     "Use_Default_Advanced_Parameters": True,
#     "trained": False,
#     "augmentation": False,
#     # "pretrained_model": False,
#     "Pretrained_model_choice": models.params.Pretrained_model_choice.MODEL_NAME,
#     "Weights_choice": models.params.Weights_choice.BEST,
#     # "QC_model_path": os.path.join(".dl4mic", "qc"),
#     "QC_model_path": "",
#     "QC_model_name": None,
#     "Multiply_dataset_by": 2,
#     "Save_augmented_images": False,
#     "Saving_path": "",
#     "Use_Default_Augmentation_Parameters": True,
#     "rotate_90_degrees": 0.5,
#     "rotate_270_degrees": 0.5,
#     "flip_left_right": 0.5,
#     "flip_top_bottom": 0.5,
#     "random_zoom": 0,
#     "random_zoom_magnification": 0.9,
#     "random_distortion": 0,
#     "image_shear": 0,
#     "max_image_shear": 10,
#     "skew_image": 0,
#     "skew_image_magnitude": 0,
# }


class CARE(models.DL4MicModelTF):

    # model: str ="CARE"
    # model_name: str = None
    # model_path: str = None
    # Notebook_version": 1.12,
    initial_learning_rate: float = 0.0004
    number_of_steps : float = 400
    number_of_patches: float = 100
    percentage_validation: int = 10
    # image_patches": None,
    # loss_function": None,
    batch_size: int = 16
    patch_size: int = 80
    # Training_source": None,
    number_of_epochs: int = 100
    Use_Default_Advanced_Parameters: bool = True
    # trained": False,
    # augmentation": False,
    #  "pretrained_model": False,
    Pretrained_model_choice: str = models.params.Pretrained_model_choice.MODEL_NAME
    Weights_choice: str = models.params.Weights_choice.BEST
    model_name: str = "care"
    network: str = "CARE 2D"
    description: str = "CARE 2D trained using ZeroCostDL4Mic."
    ref_str: str = '- CARE: Weigert, Martin, et al. "Content-aware image restoration: pushing the limits of fluorescence microscopy." Nature methods 15.12 (2018): 1090-1097.'
    # authors: List[str] = ["You"]

    #  "QC_model_path": os.path.join(".dl4mic", "qc"),
    # QC_model_path": "",
    # QC_model_name": None,


    # import N2V
    # config=None
    # self.dl4mic_model_config={}

    # def init(self):
    #     self.network = "CARE 2D"
    #     self.model_name = "CARE"
    #     self.description = "Noise2Void 2D trained using ZeroCostDL4Mic.'"
    #     self.authors = ["You"]
    #     self.ref_str = '- CARE: Weigert, Martin, et al. "Content-aware image restoration: pushing the limits of fluorescence microscopy." Nature methods 15.12 (2018): 1090-1097.'

    def get_data(self):
        (self.X_train, self.Y_train), (self.X_test, self.Y_test), self.axes = get_data(
            self.folders.Training_source,
            self.folders.Training_target,
            self.patch_size,
            "",
            self.number_of_patches,
            self.percentage_validation,
            self.folders.model_path,
        )

    def get_config(self):
        self.get_data()
        self.get_channels()
        self.config = get_care_config(
            self.X_train,
            self.Use_Default_Advanced_Parameters,
            self.batch_size,
            self.Use_pretrained_model,
            self.Weights_choice,
            self.initial_learning_rate,
            self.lastLearningRate,
            self.bestLearningRate,
            self.number_of_epochs,
            self.axes,
            self.n_channel_in,
            self.n_channel_out,
        )

    def get_channels(self):
        (self.n_channel_in, self.n_channel_out) = get_channels(
            self.X_train, self.Y_train, self.axes
        )

    def get_model(self):
        self.get_config()
        self.model = get_care_model(
            self.config,
            self.model_name,
            self.folders.model_path,
            self.Use_pretrained_model,
            self.folders.h5_file_path,
        )

    def train_model(self):
        train_model(
            self.X_train,
            self.Y_train,
            self.X_test,
            self.Y_test,
            self.model,
            self.folders.model_path,
            self.model_name,
        )

    def run(self):
        self.model = self.get_model()
        self.pre_training(self.X_train)
        self.history = self.train_model()
        self.post_training(self.history)

    def gleen_data(self,*args,**kwargs):
        self.get_channels()
        self.get_model()
        self.get_data()
        self.get_config()
    def split_data(self, Xdata):
        pass
def train_model(X, Y, X_val, Y_val, model_training, model_path, model_name):
    start = time.time()

    # Start Training
    history = model_training.train(X, Y, validation_data=(X_val, Y_val))

    print("Training, done.")

    # convert the history.history dict to a pandas DataFrame:
    lossData = pd.DataFrame(history.history)
    qc_path = os.path.join(model_path, model_name, "Quality Control")
    if os.path.exists(qc_path):
        shutil.rmtree(qc_path)

    os.makedirs(qc_path)

    # The training evaluation.csv is saved (overwrites the Files if needed).
    lossDataCSVpath = os.path.join(qc_path, "training_evaluation.csv")
    with open(lossDataCSVpath, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["loss", "val_loss", "learning rate"])
        for i in range(len(history.history["loss"])):
            writer.writerow(
                [
                    history.history["loss"][i],
                    history.history["val_loss"][i],
                    history.history["lr"][i],
                ]
            )

    # Displaying the time elapsed for training
    dt = time.time() - start
    mins, sec = divmod(dt, 60)
    hour, mins = divmod(mins, 60)
    print("Time elapsed:", hour, "hour(s)", mins, "min(s)", round(sec), "sec(s)")

    model_training.export_TF()

    print(
        "Your model has been sucessfully exported and can now also be used in the CSBdeep Fiji plugin"
    )
    return history
    # pass


def get_data(
    Training_source,
    Training_target,
    patch_size,
    base_path,
    number_of_patches,
    percentage_validation,
    model_path,
):
    percentage = percentage_validation / 100
    # def get_data():
    raw_data = data.RawData.from_folder(
        basepath=base_path,
        source_dirs=[Training_source],
        target_dir=Training_target,
        axes="CYX",
        pattern="*.tif*",
    )

    X, Y, XY_axes = data.create_patches(
        raw_data,
        patch_filter=None,
        patch_size=(patch_size, patch_size),
        n_patches_per_image=number_of_patches,
    )

    print("Creating 2D training dataset")
    training_path = os.path.join(model_path,"rawdata")
    rawdata1 = training_path + ".npz"
    np.savez(training_path, X=X, Y=Y, axes=XY_axes)

    # Load Training Data
    return load_training_data(rawdata1, validation_split=percentage, verbose=True)


def get_channels(X, Y, axes):
    c = axes_dict(axes)["C"]
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]
    return (n_channel_in, n_channel_out)


def get_care_config(
    X,
    Use_Default_Advanced_Parameters,
    batch_size,
    Use_pretrained_model,
    Weights_choice,
    initial_learning_rate,
    lastLearningRate,
    bestLearningRate,
    number_of_epochs,
    axes,
    n_channel_in,
    n_channel_out,
):
    # Here we automatically define number_of_step in function of training data and batch size

    if Use_Default_Advanced_Parameters:
        number_of_steps = int(X.shape[0] / batch_size) + 1

    # --------------------- Using pretrained model ------------------------
    # Here we ensure that the learning rate set correctly when using pre-trained models
    if Use_pretrained_model:
        if Weights_choice == "last":
            initial_learning_rate = lastLearningRate

        if Weights_choice == "best":
            initial_learning_rate = bestLearningRate
    # --------------------- ---------------------- ------------------------

    # Here we create the configuration file

    config = csbdeep.models.Config(
        axes,
        n_channel_in,
        n_channel_out,
        probabilistic=True,
        train_steps_per_epoch=number_of_steps,
        train_epochs=number_of_epochs,
        unet_kern_size=5,
        unet_n_depth=3,
        train_batch_size=batch_size,
        train_learning_rate=initial_learning_rate,
    )
    return config


def get_care_model(config, model_name, model_path, Use_pretrained_model, h5_file_path):
    model_training = csbdeep.models.CARE(config, model_name, basedir=model_path)
    # --------------------- Using pretrained model ------------------------
    # Load the pretrained weights
    if Use_pretrained_model:
        model_training.load_weights(h5_file_path)
    # --------------------- ---------------------- ------------------------
    return model_training
    # pdf_export(augmentation = Use_Data_augmentation, pretrained_model = Use_pretrained_model)
