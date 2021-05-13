import os, random
from tifffile import imread, imsave
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
import wget
import shutil
from enum import Enum
import pandas as pd
from . import checks
from . import utils
from collections.abc import Mapping

# from .utils import get_h5_path
from pathlib import Path
from . import reporting


class params:
    # class Weights_choice(Enum):
    class Weights_choice(Enum):
        BEST = "best"
        LAST = "last"

    class Pretrained_model_choice(Enum):
        MODEL_NAME = "Model_name"
        MODEL_FROM_FILE = "Model_from_file"


class DL4MicModel(Mapping):

    base_out_folder = ".dl4mic"
    model_name = "temp"
    example_image = None

    full_config = {}
    dl4mic_model_config = {
        # "model":"N2V",
        "model_name": None,
        "model_path": None,
        "ref_str": None,
        "Notebook_version": 1.12,
        "initial_learning_rate": 0.0004,
        "number_of_steps": 100,
        "percentage_validation": 10,
        "image_patches": None,
        "loss_function": None,
        "batch_size": 128,
        "patch_size": 64,
        "Training_source": None,
        "number_of_epochs": 100,
        "Use_Default_Advanced_Parameters": False,
        "trained": False,
        "augmentation": False,
        "pretrained_model": False,
        "Pretrained_model_choice": params.Pretrained_model_choice.MODEL_NAME,
        "Weights_choice": params.Weights_choice.BEST,
    }

    def __init__(self, model_config={}):
        self.output_folder = os.path.join(self.base_out_folder, self.model_name)
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

        self.dl4mic_model_config["model_path"] = os.path.join(
            self.base_out_folder, self.model_name, "model"
        )

        self.dl4mic_model_config.update(model_config)

        Path(self.dl4mic_model_config["model_path"]).mkdir(parents=True, exist_ok=True)

        self.model_specifics()
        self.interface()

    def model_specifics(self):
        pass

    def __iter__(self):
        return iter(self.dl4mic_model_config)

    def __len__(self):
        return len(self.dl4mic_model_config)

    def __getitem__(self, arg):
        return self.dl4mic_model_config[arg]

    def model_specifics(self):
        pass

    def interface(self):
        pass

    def set_model_config(self):
        pass

    def set_model_params(self):
        pass

    def check_model_params(self):
        self.check_model_specific_params()
        pass

    def check_model_specific_params(self):
        pass

    def get_ref(self):
        return self.dl4mic_model_config["ref_str"]

    def __repr__(self):
        self.dl4mic_model_config

    def append_config(self, config_dict):
        self.dl4mic_model_config.update(config_dict)
        # return self.dl4mic_model_config

    def get_config(self):
        return self.dl4mic_model_config

    def get_config_df(self):
        return pd.DataFrame(self.dl4mic_model_config)

    # def data_checks(self):
    #     self.dl4mic_model_config["patch_size"] = checks.check_image_dims(
    #         self.dl4mic_model_config["patch_size"], self.dl4mic_model_config["Training_source"]
    #     )

    def get_h5_path(self):
        h5_file_path = utils.get_h5_path(
            self.dl4mic_model_config, self.dl4mic_model_config["Weights_choice"]
        )
        h5_file_path
        self.dl4mic_model_config["h5_file_path"] = h5_file_path
        return h5_file_path

    def use_pretrained_model(self):
        pass

    def get_model_params(self):
        return self.full_config[self.model_params]

    def get_config_params(self):
        return self.full_config[self.model_config]

    def model_export_tf(self, model, X_val):
        patch_size = self.dl4mic_model_config["batch_size"]
        model.export_TF(
            name=self.model_name,
            description=self.model_description,
            authors=self.authors,
            test_img=X_val[0, ..., 0],
            axes="YX",
            patch_shape=(
                self.dl4mic_model_config["patch_size"],
                self.dl4mic_model_config["patch_size"],
            ),
        )

    def data_checks(self, show_image=False):
        # checks.check_for_prexisiting_model()

        image = checks.get_random_image(self.dl4mic_model_config["Training_source"])

        checks.check_data(image)

        filename = os.path.join(self.output_folder, "TrainingDataExample.png")
        if show_image:
            checks.display_image(image, filename)

        checks.check_image_dims(image, self.dl4mic_model_config["patch_size"])

        return image

    def data_augmentation(self):
        pass

    def load_pretrained_model(self):
        if self.dl4mic_model_config["Use_pretrained_model"]:

            self.h5_file_path = utils.download_model(
                self.dl4mic_model_config["pretrained_model_path"],
                self.dl4mic_model_config["pretrained_model_choice"],
                self.dl4mic_model_config["pretrained_model_name"],
                self.dl4mic_model_config["Weights_choice"],
            )

            learning_rates_dict = utils.load_model(
                self.dl4mic_model_config["h5_file_path"],
                self.dl4mic_model_config["pretrained_model_path"],
                self.dl4mic_model_config["Weights_choice"],
                self.dl4mic_model_config["initial_learning_rate"],
            )

            self.append_config(learning_rates_dict)
            return self.h5_file_path
        else:
            pass

    def report(self, time_start=None):

        report_args = [
            "model_name",
            "model_path",
            "ref_str",
            "Notebook_version",
            "initial_learning_rate",
            "number_of_steps",
            "percentage_validation",
            "image_patches",
            "loss_function",
            "batch_size",
            "patch_size",
            "Training_source",
            "number_of_epochs",
            # "time_start",
            "Use_Default_Advanced_Parameters",
            "trained",
            "augmentation",
            "pretrained_model",
        ]
        extra_args = {"time_start": time_start, "example_image": self.example_image}

        report_config = {key: self.dl4mic_model_config[key] for key in report_args}
        report_config.update(extra_args)

        return reporting.pdf_export(**report_config)


class N2V(DL4MicModel):
    # import N2V
    # config=None
    # self.dl4mic_model_config={}
    model_name = "N2V"
    description = "Noise2Void 2D trained using ZeroCostDL4Mic.'"
    authors = ["You"]
    ref_str = '- Noise2Void: Krull, Alexander, Tim-Oliver Buchholz, and Florian Jug. "Noise2void-learning denoising from single noisy images." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.'

    def set_model_config(self):
        self.model_config = [
            "train_steps_per_epoch",
            "train_epochs",
            "train_batch_size",
        ]

    def set_model_params(self):
        self.model_params = ["model_name", "model_path"]

    def interface(self):
        # self.full_config = self.dl4mic_model_config
        interface_dict = {
            "name": self.dl4mic_model_config["model_name"],
            "basedir": self.dl4mic_model_config["model_path"],
            "train_steps_per_epoch": self.dl4mic_model_config["number_of_steps"],
            "train_epochs": self.dl4mic_model_config["number_of_epochs"],
            "train_batch_size": self.dl4mic_model_config["batch_size"],
            "directory": self.dl4mic_model_config["Training_source"],
        }
        self.dl4mic_model_config.update(interface_dict)

    def model_specifics(self):
        specifics = {
            "model_name": self.model_name,
            "description": self.description,
            "authors": self.authors,
            "ref_str": self.ref_str,
        }
        self.dl4mic_model_config.update(specifics)

    def gleen_data(self, Xdata):
        shape_of_Xdata = Xdata.shape

        self.dl4mic_model_config["shape_of_Xdata"] = shape_of_Xdata

        self.get_threshold(self.dl4mic_model_config["shape_of_Xdata"])
        self.get_image_patches(self.dl4mic_model_config["shape_of_Xdata"])
        if self.dl4mic_model_config["Use_Default_Advanced_Parameters"]:
            self.dl4mic_model_config(self.dl4mic_model_config["shape_of_Xdata"])

    def get_threshold(self, shape_of_Xdata):
        self.dl4mic_model_config["threshold"] = int(
            shape_of_Xdata[0]
            * (self.dl4mic_model_config["percentage_validation"] / 100)
        )
        return self.dl4mic_model_config["threshold"]

    def get_image_patches(self, shape_of_Xdata):
        self.dl4mic_model_config["image_patches"] = int(shape_of_Xdata[0])
        return self.dl4mic_model_config["image_patches"]

    def get_default_steps(self, shape_of_Xdata):
        self.dl4mic_model_config["number_of_steps"] = (
            int(shape_of_Xdata / self.dl4mic_model_config["batch_size"]) + 1
        )
        return self.dl4mic_model_config["number_of_steps"]


# class N2V():
#     # import N2V
#     # config=None
#     # self.dl4mic_model_config={}
#     def __init__(self):
#         self.dl4mic_model_config = {
#             # "model":"N2V",
#             "model_name": None,
#             "model_path": None,
#             # "ref_str"=,
#             "Notebook_version": 1.12,
#             "initial_learning_rate": 0.0004,
#             "number_of_steps": 100,
#             "percentage_validation": 10,
#             # "image_patches"=,
#             # "loss_function"=,
#             "batch_size": 128,
#             "patch_size": 64,
#             "Training_source": None,
#             "number_of_epochs": 100,
#             "Use_Default_Advanced_Parameters": False,
#             "trained": False,
#             "augmentation": False,
#             "pretrained_model": False,
#             "Pretrained_model_choice": params.Pretrained_model_choice.Model_name,
#             "Weights_choice": params.Pretrained_model_choice.best,
#         }
#         self.model_specifics()

#     def set_model_config(self):
#         self.model_config = ["train_steps_per_epoch","train_epochs","train_batch_size"]
#     def set_model_params(self):
#         self.model_params = ["model_name","model_path"]

#     # def __init__():
#     # datagen = N2V_DataGenerator()
#     # return
#     def get_ref(self):
#         return self.dl4mic_model_config["ref_str"]

#     def __getitem__(self, arg):
#         return self.dl4mic_model_config[arg]

#     def append_config(self, config_dict):
#         self.dl4mic_model_config = self.dl4mic_model_config.update(config_dict)
#         return self.dl4mic_model_config

#     def get_config(self):
#         # dl4mic_model_config = {
#         #                 "image_patches" = None}
#         # dl4mic_model_config = {"image_patches"=1}
#         # Xdata.shape[0],
#         # "loss_function" = config.train_loss
#         return self.dl4mic_model_config

#     def data_checks(self):
#         self.dl4mic_model_config["patch_size"] = checks.check_image_dims(
#             self.dl4mic_model_config["patch_size"], self.dl4mic_model_config["Training_source"]
#         )

#     def get_h5_path(self):
#         self.dl4mic_model_config["h5_file_path"] = os.path.join(
#             self.dl4mic_model_config["pretrained_model_path"],
#             "weights_" + self.dl4mic_model_config["Weights_choice"] + ".h5",
#         )

#     def use_pretrained_model(self):
#         pass

#     def interface(self):
#         self.full_config = self.dl4mic_model_config
#         interface_dict = {
#             "name":self.dl4mic_model_config["model_name"],
#             "basedir": self.dl4mic_model_config["model_path"],
#             "train_steps_per_epoch":self.dl4mic_model_config["number_of_steps"],
#             "train_epochs":self.dl4mic_model_config["number_of_epochs"],
#             "train_batch_size":self.dl4mic_model_config["batch_size"],
#         }
#         self.N2V_config.update(interface_dict)

#     def get_model_params(self):
#         return self.full_config[self.model_params]

#     def get_config_params(self):
#         return self.full_config[self.model_config]
#         # self.N2V_config["name"] = self.dl4mic_model_config["model_name"]
#         # self.N2V_config["basedir"] = self.dl4mic_model_config["model_path"]
#         # self.N2V_config["basedir"] = self.dl4mic_model_config["model_path"]
#     def model_export_tf(self,model,X_val):
#         patch_size = self.dl4mic_model_config["batch_size"]
#         model.export_TF(
#                 name=self.model_name,
#                 description=self.model_description,
#                 authors=self.authors,
#                 test_img=X_val[0,...,0], axes='YX',
#                 patch_shape=(self.dl4mic_model_config["patch_size"],
#                              self.dl4mic_model_config["patch_size"]))
#     def model_specifics(self):
#         self.model_name = "N2V"
#         self.description = "Noise2Void 2D trained using ZeroCostDL4Mic.'"
#         self.authors = ["You"]
