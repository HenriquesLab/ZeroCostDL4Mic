# from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
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
from csbdeep.models import Config, CARE
from csbdeep import data


# def __init__(self):
#     return self.N2V

# from models import params

default_params = {
    "model": "CARE",
    "model_name": None,
    "model_path": None,
    "ref_str": None,
    "Notebook_version": 1.12,
    "initial_learning_rate": 0.0004,
    "number_of_steps": 400,
    "number_of_patches": 100,
    "percentage_validation": 10,
    "image_patches": None,
    "loss_function": None,
    "batch_size": 16,
    "patch_size": 80,
    "Training_source": None,
    "number_of_epochs": 100,
    "Use_Default_Advanced_Parameters": True,
    "trained": False,
    "augmentation": False,
    "pretrained_model": False,
    "Pretrained_model_choice": models.params.Pretrained_model_choice.MODEL_NAME,
    "Weights_choice": models.params.Weights_choice.BEST,
    # "QC_model_path": os.path.join(".dl4mic", "qc"),
    "QC_model_path": "",
    "QC_model_name": None,
    "Multiply_dataset_by": 2,
    "Save_augmented_images": False,
    "Saving_path": "",
    "Use_Default_Augmentation_Parameters": True,
    "rotate_90_degrees": 0.5,
    "rotate_270_degrees": 0.5,
    "flip_left_right": 0.5,
    "flip_top_bottom": 0.5,
    "random_zoom": 0,
    "random_zoom_magnification": 0.9,
    "random_distortion": 0,
    "image_shear": 0,
    "max_image_shear": 10,
    "skew_image": 0,
    "skew_image_magnitude": 0,
}


class CARE(models.DL4MicModelTF):

    # import N2V
    # config=None
    # self.dl4mic_model_config={}

    def init(self):
        self.network = "CARE 2D"
        self.model_name = "CARE"
        self.description = "Noise2Void 2D trained using ZeroCostDL4Mic.'"
        self.authors = ["You"]
        self.ref_str = '- CARE: Weigert, Martin, et al. "Content-aware image restoration: pushing the limits of fluorescence microscopy." Nature methods 15.12 (2018): 1090-1097.'
        
    def run(self):
        pass

    def get_model(self):

        # --------------------- Here we load the augmented data or the raw data ------------------------

        if self.Use_Data_augmentation:
            Training_source_dir = self.Training_source_augmented
            Training_target_dir = self.Training_target_augmented

        if not self.Use_Data_augmentation:
            Training_source_dir = self.Training_source
            Training_target_dir = self.Training_target
        # --------------------- ------------------------------------------------

        # This object holds the image pairs (GT and low), ensuring that CARE compares corresponding images.
        # This file is saved in .npz format and later called when loading the trainig data.

        raw_data = data.RawData.from_folder(
            basepath=self.base_out_folder,
            source_dirs=[Training_source_dir],
            target_dir=Training_target_dir,
            axes="CYX",
            pattern="*.tif*",
        )

        X, Y, XY_axes = data.create_patches(
            raw_data,
            patch_filter=None,
            patch_size=(self.patch_size, self.patch_size),
            n_patches_per_image=self.number_of_patches,
        )

        print("Creating 2D training dataset")
        training_path = self.model_path + "/rawdata"
        rawdata1 = training_path + ".npz"
        np.savez(training_path, X=X, Y=Y, axes=XY_axes)

        # Load Training Data
        (X, Y), (X_val, Y_val), axes = load_training_data(
            rawdata1, validation_split=self.percentage / 100, verbose=True
        )
        c = axes_dict(axes)["C"]
        n_channel_in, n_channel_out = X.shape[c], Y.shape[c]
