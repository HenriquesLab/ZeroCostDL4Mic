from .. import predict, quality, checks, utils, prepare, reporting, assess
import os, random
from tifffile import imread, imsave
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
import wget
import shutil
from enum import Enum
import pandas as pd
import time

from collections.abc import Mapping

# from .utils import get_h5_path
from pathlib import Path


class params:
    # class Weights_choice(Enum):
    class Weights_choice(Enum):
        BEST = "best"
        LAST = "last"

    class Pretrained_model_choice(Enum):
        MODEL_NAME = "Model_name"
        MODEL_FROM_FILE = "Model_from_file"

    class Data_type(Enum):
        SINGLE_IMAGES = "Single_Images"
        STACKS = "Stacks"

    def get_defaults():
        # default_params():
        return {
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
            # "QC_model_path": os.path.join(".dl4mic", "qc"),
            "QC_model_path": "",
            "QC_model_name": None,
        }


# if (Use_Default_Advanced_Parameters):
#   print("Default advanced parameters enabled")
#   # number_of_steps is defined in the following cell in this case
#   batch_size = 128
#   percentage_validation = 10
#   initial_learning_rate = 0.0004


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
        "ref_aug" : '- Augmentor: Bloice, Marcus D., Christof Stocker, and Andreas Holzinger. "Augmentor: an image augmentation library for machine learning." arXiv preprint arXiv:1708.04680 (2017).',
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
        # "QC_model_path": os.path.join(".dl4mic", "qc"),
        "QC_model_path": "",
        "QC_model_name": None,
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
    def import_checks(self):
        pass

    def __iter__(self):
        return iter(self.dl4mic_model_config)

    def __len__(self):
        return len(self.dl4mic_model_config)

    def __getitem__(self, arg):
        # return getattr(self,arg) #Move away from bloody dict
        return self.dl4mic_model_config[arg]

    def __setitem__(self, key, value):
        self.dl4mic_model_config[key] = value
        # return

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

        # image = checks.get_random_image(self.dl4mic_model_config[""])
        # Training_source = self.dl4mic_model_config["Training_source"]
        Training_source = self.dl4mic_model_config["Training_source"]
        output_folder = self.output_folder
        patch_size = self.dl4mic_model_config["patch_size"]

        # checks.check_data(image)

        # filename = os.path.join(self.output_folder, "TrainingDataExample.png")
        # if show_image:
        # checks.display_image(image, filename)

        # checks.check_image_dims(image, self.dl4mic_model_config["patch_size"])

        return checks.full(Training_source, output_folder, patch_size, show_image)

    def data_augmentation(self):
        pass

    def load_pretrained_model(self):
        if self.dl4mic_model_config["Use_pretrained_model"]:

            self.dl4mic_model_config["h5_file_path"] = utils.download_model(
                self.dl4mic_model_config["pretrained_model_path"],
                self.dl4mic_model_config["pretrained_model_choice"],
                self.dl4mic_model_config["pretrained_model_name"],
                self.dl4mic_model_config["Weights_choice"],
                self.dl4mic_model_config["model_path"],
            )

            learning_rates_dict = utils.load_model(
                self.dl4mic_model_config["h5_file_path"],
                self.dl4mic_model_config["pretrained_model_path"],
                self.dl4mic_model_config["Weights_choice"],
                self.dl4mic_model_config["initial_learning_rate"],
            )

            self.append_config(learning_rates_dict)
            return self.dl4mic_model_config["h5_file_path"]
        else:
            pass

    def reporting(self):
        pass

    def report(self, time_start=None, trained=None, show_image=False):

        report_args = [
            "model_name",
            "model_path",
            "ref_str",
            "ref_aug",
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
            # "trained",
            "augmentation",
            "pretrained_model",
        ]
        extra_args = {
            "time_start": time_start,
            "example_image": self.example_image,
            "trained": trained,
        }

        report_config = {key: self.dl4mic_model_config[key] for key in report_args}
        report_config.update(extra_args)

        return reporting.pdf_export(**report_config)

    def pre_report(
        self,
        X_train=None,
        X_test=None,
        time_start=None,
        trained=False,
        show_image=False,
    ):
        if show_image:
            prepare.setup_complete(X_train=X_train, X_test=X_test)
        return self.report(time_start=time_start, trained=None, show_image=False)

    def post_report(
        self, X_train=None, X_test=None, time_start=None, trained=True, show_image=False
    ):
        return self.report(
            time_start=time_start, trained=trained, show_image=show_image
        )

    # def quality_stock(self):
    #     # Path(self.dl4mic_model_config["QC_model_path"]).mkdir(parents=True, exist_ok=True)

    #     return quality.quality_sequence(
    #         model_path,
    #         model_name,
    #         QC_model_name,
    #         QC_model_path,
    #         ref_str,
    #         network,
    #         Use_the_current_trained_model,
    #         Source_QC_folder,
    #         Target_QC_folder,
    #     )
    def quality_extra(self, **kwargs):
        pass

    def quality(self, history=None):

        model_path = self.dl4mic_model_config["model_path"]
        model_name = self.dl4mic_model_config["model_name"]

        if self.dl4mic_model_config["QC_model_name"] is None:
            self.dl4mic_model_config["QC_model_name"] = model_name

        if self.dl4mic_model_config["QC_model_path"] is None:
            self.dl4mic_model_config["QC_model_path"] = model_path

        QC_model_name = self.dl4mic_model_config["QC_model_name"]
        QC_model_path = self.dl4mic_model_config["QC_model_path"]

        ref_str = self.dl4mic_model_config["ref_str"]
        network = self.dl4mic_model_config["network"]
        Use_the_current_trained_model = self.dl4mic_model_config[
            "Use_the_current_trained_model"
        ]
        Source_QC_folder = self.dl4mic_model_config["Source_QC_folder"]
        Target_QC_folder = self.dl4mic_model_config["Target_QC_folder"]
        self.QC_dir = QC_model_path + QC_model_name
        Path(self.QC_dir).mkdir(parents=True, exist_ok=True)

        # return self.quality_stock()
        # def quality(self):

        if history != None:
            self.quality_extra(history=history)

        return quality.full(
            model_path,
            model_name,
            QC_model_name,
            QC_model_path,
            ref_str,
            network,
            Use_the_current_trained_model,
            Source_QC_folder,
            Target_QC_folder,
        )

    def predict(self):

        Prediction_model_path = self.dl4mic_model_config["Prediction_model_path"]
        Prediction_model_name = self.dl4mic_model_config["Prediction_model_name"]

        return predict.full(Prediction_model_path, Prediction_model_name)

    def assess(self):

        Prediction_model_path = self.dl4mic_model_config["Prediction_model_path"]
        Prediction_model_name = self.dl4mic_model_config["Prediction_model_name"]
        Data_type = self.dl4mic_model_config["Data_type"]

        return assess.full(Prediction_model_path, Prediction_model_name, Data_type)

    def save_model(self):
        pass

    def get_model(self, **kwargs):
        pass

    def run(self, config):
        pass

    def pre_training(self, X):

        # if data_checks.__name__ == self.__class__
        self.data_checks()
        # self.data_checks_specific() #Be smarter with class inheritence

        self.data_augmentation()
        # self.data_augmentation_specific()

        self.gleen_data(X)
        self.split_data(X)
        self.check_model_params()
        pdf = self.pre_report(
            X_train=self.dl4mic_model_config["X_train"],
            X_test=self.dl4mic_model_config["X_test"],
            show_image=False,
        )
        self.pre_training_specific()
        self.check_model_params()
        return pdf

    def pre_training_specific(self):
        pass

    def post_training(self, history=None, show_image=False):
        self.post_training_specific()
        self.quality(history)
        pdf = self.post_report(show_image)
        self.predict()
        self.assess()
        return pdf

    def post_training_specific(self):
        pass

    def split_data(self, Xdata):
        threshold = self.dl4mic_model_config["threshold"]
        X = Xdata[threshold:]
        X_val = Xdata[:threshold]
        self.dl4mic_model_config["X_train"] = X
        self.dl4mic_model_config["X_test"] = X_val
        return X, X_val
    
    # def default_augment(self):
    #     Use_Default_Augmentation_Parameters = self.dl4mic_model_config["Use_Default_Augmentation_Parameters"]

    #     if Use_Default_Augmentation_Parameters:
    #         rotate_90_degrees = 0.5
    #         rotate_270_degrees = 0.5
    #         flip_left_right = 0.5
    #         flip_top_bottom = 0.5

    #         if not Multiply_dataset_by >5:
    #             random_zoom = 0
    #             random_zoom_magnification = 0.9
    #             random_distortion = 0
    #             image_shear = 0
    #             max_image_shear = 10
    #             skew_image = 0
    #             skew_image_magnitude = 0

    #         if Multiply_dataset_by >5:
    #             random_zoom = 0.1
    #             random_zoom_magnification = 0.9
    #             random_distortion = 0.5
    #             image_shear = 0.2
    #             max_image_shear = 5
    #             skew_image = 0.2
    #             skew_image_magnitude = 0.4

    #         if Multiply_dataset_by >25:
    #             random_zoom = 0.5
    #             random_zoom_magnification = 0.8
    #             random_distortion = 0.5
    #             image_shear = 0.5
    #             max_image_shear = 20
    #             skew_image = 0.5
    #             skew_image_magnitude = 0.6


    # def quality_tf(self, model, model_path, model_name,QC_model_name,QC_model_path):
    #     df = self.get_history_df_from_model_tf(model)
    #     quality.df_to_csv(df, model_path, model_name)
    #     quality.display_training_errors(model, QC_model_name, QC_model_path)

    #     return df
    # model_path = self.dl4mic_model_config["model_path"]
    # model_name = self.dl4mic_model_config["model_name"]

    # QC_model_name = self.dl4mic_model_config["QC_model_name"]
    # QC_model_path = self.dl4mic_model_config["QC_model_path"]

    # Source_QC_folder = self.dl4mic_model_config["Source_QC_folder"]
    # Target_QC_folder = self.dl4mic_model_config["Target_QC_folder"]
    # def quality_sequence(self,model_path,model_name,QC_model_name,QC_model_path):

    #     Use_the_current_trained_model = self.dl4mic_model_config[
    #         "Use_the_current_trained_model"
    #     ]
    #     # quality_tf(self, model, model_path, model_name)
    #     quality.quality_folder_reset(model_path, model_name)
    #     quality.qc_model_checks(
    #         QC_model_name,
    #         QC_model_path,
    #         model_name,
    #         model_path,
    #         Use_the_current_trained_model,
    #     )

    #     reporting.qc_pdf_export()
    # self.post_report()

    # def get_history_df_from_model_tf(self, model):
    #     history = model.history
    #     return pd.DataFrame(history.history)


class DL4MicModelTF(DL4MicModel):
    def save_model(self, model, X_val):
        patch_size = self.dl4mic_model_config["patch_size"]
        model.export_TF(
            name=self.model_name,
            description=self.description,
            authors=self.authors,
            test_img=X_val[0, ..., 0],
            axes="YX",
            patch_shape=(patch_size, patch_size),
        )
        print(
            "Your model has been sucessfully exported and can now also be used in the CSBdeep Fiji plugin"
        )

    def history_to_df(history):
        return pd.DataFrame(history.history)

    def quality_checks(self, history):
        pass

    # def quality(self, history):
    #     if self.dl4mic_model_config["Use_the_current_trained_model"]:
    #         self.dl4mic_model_config["QC_model_path"] = self.dl4mic_model_config[
    #             "model_path"
    #         ]
    #         self.dl4mic_model_config["QC_model_name"] = self.dl4mic_model_config[
    #             "model_name"
    #         ]
    #     # model = self.dl4mic_model_config[""

    #     model_path = self.dl4mic_model_config["model_path"]
    #     model_name = self.dl4mic_model_config["model_name"]
    #     QC_model_name = self.dl4mic_model_config["QC_model_name"]
    #     QC_model_path = self.dl4mic_model_config["QC_model_path"]

    #     qc_folder = os.path.join(model_path, model_name, "Quality Control")

    #     quality.quality_tf(
    #         history, model_path, model_name, QC_model_name, QC_model_path
    #     )

    # return self.quality_stock()


from .N2V import N2V
from .CARE import CARE


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
