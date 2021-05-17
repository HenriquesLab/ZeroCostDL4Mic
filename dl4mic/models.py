from . import predict, quality, checks, utils, prepare, reporting, assess
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

    def get_model(self,**kwargs):
        pass

    def run(self,config):
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

    def post_training(self,history=None,show_image=False):
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
        return X,X_val


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


class N2V(DL4MicModelTF):
    # import N2V
    # config=None
    # self.dl4mic_model_config={}
    network = "Noise2Void"
    model_name = "n2v"
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
            "network": self.network,
        }
        self.dl4mic_model_config.update(specifics)

    def gleen_data(self, Xdata):
        shape_of_Xdata = Xdata.shape

        self.dl4mic_model_config["shape_of_Xdata"] = shape_of_Xdata

        self.get_threshold(self.dl4mic_model_config["shape_of_Xdata"])
        self.get_image_patches(self.dl4mic_model_config["shape_of_Xdata"])
        if self.dl4mic_model_config["Use_Default_Advanced_Parameters"]:
            self.dl4mic_model_config["number_of_steps"] = self.get_default_steps(self.dl4mic_model_config["shape_of_Xdata"])
            
           
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
            int(shape_of_Xdata[0] / self.dl4mic_model_config["batch_size"]) + 1
        )
        return self.dl4mic_model_config["number_of_steps"]

    def save_model(model):
        pass

    def quality_extra(self, history):
        # history = self.dl4mic_model_config["history"]
        model_path = self.dl4mic_model_config["model_path"]
        model_name = self.dl4mic_model_config["model_name"]
        QC_model_name = self.dl4mic_model_config["QC_model_name"]
        QC_model_path = self.dl4mic_model_config["QC_model_path"]

        quality.quality_tf(
            history, model_path, model_name, QC_model_name, QC_model_path
        )

    def get_model(self):

        dl4mic_model = self.dl4mic_model_config
        # def n2v_get_model(dl4mic_model, Xdata):

            ################ N2V ######################

        from n2v.models import N2VConfig, N2V
        from csbdeep.utils import plot_history
        from n2v.utils.n2v_utils import manipulate_val_data
        from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
        from csbdeep.io import save_tiff_imagej_compatible

        threshold = dl4mic_model["threshold"]
        image_patches = dl4mic_model["image_patches"]
        shape_of_Xdata = dl4mic_model["shape_of_Xdata"]

        print(shape_of_Xdata[0], "patches created.")
        print(
            dl4mic_model["threshold"],
            "patch images for validation (",
            dl4mic_model["percentage_validation"],
            "%).",
        )
        print(image_patches - threshold, "patch images for training.")

        config = N2VConfig(
            dl4mic_model["X_train"],
            unet_kern_size=3,
            train_steps_per_epoch=dl4mic_model["number_of_steps"],
            train_epochs=dl4mic_model["number_of_epochs"],
            train_loss=dl4mic_model["loss_function"],
            batch_norm=True,
            train_batch_size=dl4mic_model["batch_size"],
            n2v_perc_pix=0.198,
            n2v_manipulator="uniform_withCP",
            n2v_neighborhood_radius=5,
            train_learning_rate=dl4mic_model["initial_learning_rate"],
        )

        model = N2V(
            config=config,
            name=dl4mic_model["model_name"],
            basedir="tests",
        )

        print("Setup done.")
        print(config)
        return model

    def run(self):
            # import os

        os.environ["KERAS_BACKEND"] = "tensorflow"

        from n2v.internals.N2V_DataGenerator import N2V_DataGenerator

        dl4mic_model = self.dl4mic_model_config

        datagen = N2V_DataGenerator()
        imgs = datagen.load_imgs_from_directory(directory=dl4mic_model["Training_source"])

        Xdata = datagen.generate_patches_from_list(
            imgs,
            shape=(dl4mic_model["patch_size"], dl4mic_model["patch_size"]),
            augment=dl4mic_model["Use_Data_augmentation"],
        )

        self.pre_training(Xdata)

        dl4mic_model["start"] = time.time()

        # TF1 Hack
        import tensorflow.compat.v1 as tf

        tf.disable_v2_behavior()
        tf.__version__ = 1.14

        model = self.get_model()
        threshold = dl4mic_model["threshold"]

        X = Xdata[threshold:]
        X_val = Xdata[:threshold]

        history = model.train(X, X_val)
        print("Training done.")

        pdf_post = self.post_report(history)
        return self



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
