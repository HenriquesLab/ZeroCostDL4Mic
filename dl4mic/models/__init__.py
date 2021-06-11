import numpy as np
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

from mashumaro import DataClassDictMixin
from collections.abc import Mapping

from pathlib import Path
from dataclasses import dataclass

from typing import List

class params:
    class Weights_choice(Enum):
        BEST = "best"
        LAST = "last"

    class Pretrained_model_choice(Enum):
        MODEL_NAME = "Model_name"
        MODEL_FROM_FILE = "Model_from_file"

    class Data_type(Enum):
        SINGLE_IMAGES = "Single_Images"
        STACKS = "Stacks"

    # Defaults should be loaded in per submodule
    # def get_defaults():
    #     # default_params():
    #     return {
    #         # "model":"N2V",
    #         "model_name": None,
    #         "model_path": None,
    #         "ref_str": None,
    #         "Notebook_version": 1.12,
    #         "initial_learning_rate": 0.0004,
    #         "number_of_steps": 100,
    #         "percentage_validation": 10,
    #         "image_patches": None,
    #         "loss_function": None,
    #         "batch_size": 128,
    #         "patch_size": 64,
    #         "Training_source": None,
    #         "number_of_epochs": 100,
    #         "Use_Default_Advanced_Parameters": False,
    #         "trained": False,
    #         "augmentation": False,
    #         # "pretrained_model": False,
    #         "Pretrained_model_choice": params.Pretrained_model_choice.MODEL_NAME,
    #         "Weights_choice": params.Weights_choice.BEST,
    #         # "QC_model_path": os.path.join(".dl4mic", "qc"),
    #         "QC_model_path": "",
    #         "QC_model_name": None,
    #     }


# if (Use_Default_Advanced_Parameters):
#   print("Default advanced parameters enabled")
#   # number_of_steps is defined in the following cell in this case
#   batch_size = 128
#   percentage_validation = 10
#   initial_learning_rate = 0.0004


class DictLike(object):
    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, arg):
        # return getattr(self,arg) #Move away from bloody dict
        return getattr(self, arg)

    def __setitem__(self, key, value):
        setattr(self, key, value)
        # return

    pass


@dataclass
class Folders(DataClassDictMixin, DictLike):
    """
    Extends DataClassDictMixin and DictLike (probably better alternative
    availiable) so that it can be initialised with a dict easy
    """

    # model_name: str
    base_out_folder: str = ".dl4mic"
    output_folder: str = base_out_folder
    QC_model_path: str = None
    Training_source: str = None
    Training_target: str = None
    model_path: str = None
    pretrained_model_path: str = None
    Source_QC_folder: str = None
    Target_QC_folder: str = None
    Prediction_model_folder: str = None
    Prediction_model_path: str = None
    Data_folder: str = None
    h5_file_path: str = None
    Saving_path: str = None

    def __post_init__(self):
        defaults = {
            "QC_model_path": "qc",
            "Training_source": "training",
            "Training_target": "target",
            "model_path": "model",
            "pretrained_model_path": "pretrained_model",
            "Prediction_model_path": "prediction_model",
            "Source_QC_folder": "qc_source",
            "Target_QC_folder": "qc_target",
            "Prediction_model_folder": "pred",
            "Data_folder": "data",
            "h5_file_path": "weights",
            "Saving_path": "augment"
        }
        for key in defaults:
            if self[key] is None:
                self[key] = Path(os.path.join(self.output_folder, defaults[key]))
                self[key].mkdir(parents=True, exist_ok=True)

        # self.QC_model_path = os.path.join(output_folder, "qc")
        # self.Training_source = os.path.join(output_folder, "training")
        # self.Training_target= os.path.join(output_folder, "target")
        # self.model_path = os.path.join(output_folder, "model")
        # self.pretrained_model_path = os.path.join(output_folder, "pretrained_model")
        # self.Source_QC_folder = os.path.join(output_folder, "qc_source")
        # self.Target_QC_folder = os.path.join(output_folder, "qc_target")
        # self.Prediction_model_folder = os.path.join(output_folder, "pred")
        # self.Data_folder = os.path.join(output_folder, "data")
        # self.h5_file_path = os.path.join(output_folder, "weights")

    #     # self.model_name = model_name
    #     self.output_folder = os.path.join(self.base_out_folder, self.model_name)
    #     self.QC_model_path = os.path.join(self.output_folder, "qc")
    #     self.Training_source = os.path.join(self.output_folder, "training")
    #     self.Training_target = os.path.join(self.output_folder, "target")
    #     self.model_path = os.path.join(self.output_folder, "model")
    #     self.pretrained_model_path = os.path.join(self.output_folder, "pretrained_model")
    #     self.Source_QC_folder = os.path.join(self.output_folder, "qc_source")
    #     self.Target_QC_folder = os.path.join(self.output_folder, "qc_target")
    #     self.Prediction_model_folder = os.path.join(self.output_folder, "pred")
    #     self.Data_folder = os.path.join(self.output_folder, "data")
    #     self.h5_file_path = os.path.join(self.output_folder, "weights")


@dataclass
class DL4MicModelParams(DataClassDictMixin, DictLike):
    # folders: dataclass
    # folders.base_out_folder: str = ".dl4mic"
    # X_train: np.array = None
    # X_test: np.array = None
    # example_image: np.array = None
    # TODO make all of these None type and then default in submodule
    # May have solved this?
    # folders: Folders = Folders()
    model_name: str = "temp"
    folders: Folders = Folders()
    model: str = "dl4mic"
    image_patches: int = 100
    ref_str: str = "ref"
    loss_function: str = "loss"
    pretrained_model_choice: bool = False
    Use_pretrained_model: bool = False
    Use_the_current_trained_model: bool = False
    Use_Data_augmentation: bool = False
    Notebook_version: float = 1.12
    initial_learning_rate: float = 0.0004
    number_of_steps: int = 100
    number_of_patches: int = 100
    percentage_validation: int = 10
    batch_size: int = 128
    patch_size: int = 64
    number_of_epochs: int = 100
    Use_Default_Advanced_Parameters: bool = False
    trained: bool = False
    augmentation: bool = False
    # pretrained_model: bool = False
    Pretrained_model_choice: str = params.Pretrained_model_choice.MODEL_NAME
    Weights_choice: str = params.Weights_choice.BEST
    base_out_folder: str = ".dl4mic"
    # QC_model_path: str = os.path.join(base_out_folder, "qc")
    # Training_source: str = os.path.join(base_out_folder, "training")
    # Training_target: str = os.path.join(base_out_folder, "target")
    # model_path: str = base_out_folder
    # pretrained_model_path: str = os.path.join(base_out_folder, "model")
    pretrained_model_name: str = "model"
    Source_QC_folder: str = None
    Target_QC_folder: str = None
    # Prediction_model_folder: str = os.path.join(base_out_folder, "pred")
    Prediction_model_name: str = "pred"
    # Prediction_model_path: str = Prediction_model_folder
    QC_model_name: str = None
    Data_type: str = ""
    ref_aug: str = str(
        '- Augmentor: Bloice, Marcus D., Christof Stocker,'
        'and Andreas Holzinger. "Augmentor: an image augmentation '
        'library for machine learning." arXiv '
        'preprint arXiv:1708.04680 (2017).'
    )

    bestLearningRate: float = initial_learning_rate
    lastLearningRate: float = initial_learning_rate
    Multiply_dataset_by: int = 2
    Save_augmented_images: bool = False
    Use_Default_Augmentation_Parameters: bool = True
    rotate_90_degrees: str = 0.5
    rotate_270_degrees: str = 0.5
    flip_left_right: str = 0.5
    flip_top_bottom: str = 0.5
    random_zoom: str = 0
    random_zoom_magnification: str = 0.9
    random_distortion: str = 0
    image_shear: str = 0
    max_image_shear: str = 10
    skew_image: str = 0
    skew_image_magnitude: str = 0

    def __post_init__(self):
        # pass
        self.folders.output_folder = os.path.join(self.base_out_folder, self.model_name)
        self.folders.QC_dir = Path(os.path.join(self.QC_model_path, self.QC_model_name))
        self.folders.__post__init__()
        # self.folders.output_folder = self.output_folder

    # def __init__(self,*args,**kwargs):
    #     super().__init__()
    #     from_dict(self,kwargs)
    # super().__init__(**model_config)
    # h5_file_path: str = None
    # output_folder: str = os.path.join(base_out_folder, model_name)
    # folders : object = Folders(model_name)

    # folder_list: list = [
    #         "base_out_folder",
    #         "QC_model_path",
    #         "Training_source",
    #         "Training_target",
    #         "model_path",
    #         "pretrained_model_path",
    #         "pretrained_model_name",
    #         "Source_QC_folder",
    #         "Target_QC_folder",
    #         "Prediction_model_folder",
    #         "Prediction_model_path",
    #         "Data_folder",
    #         "output_folder"
    #     ]
    # def __init__(self,model_config={}):
    #     super().__init__(model_config)


# DL4MicModelParams = from_dict(data_class=B, data=data)


class DL4MicModel(DL4MicModelParams):

    # @dataclass
    class data(DictLike):
        example_image: np.array = None
        X_train: np.array = None
        Y_train: np.array = None
        X_test: np.array = None
        Y_test: np.array = None
        time_start: float = None
        trained: bool = False
        history: np.array = None
    def __post_init__(self):
        # super().__init__(**model_config)
        self.init()
        self.paths_and_dirs()
        self.model_specifics()
        # self.dl4mic_model_config.update(model_config)

        self.interface()

    def paths_and_dirs(self):
        # self.output_folder = os.path.join(self.base_out_folder, self.model_name)

        # Path(self.output_folder).mkdir(parents=True, exist_ok=True)

        # folder_dict = {k: self.__dict__[k] for k in self.folder_list}
        # folder_dict = self.folders.__dict__
        self.append_config(utils.make_folders(self.folders.__dict__))

    def init(self):
        self.authors =  ["You"]
        pass

    def step_3(self):
        self.step_3_1()
        self.step_3_2()
        pass
    def step_3_1(self):
        self.checks()
        pass
    def step_3_2(self):
        '''
        Data augmentation
        '''
        self.augmentation()
        pass
    def step_3_3(self):
        '''
        Load pretrained model
        '''
        self.load_pretrained_model()
        pass


    def step_4(self):
        '''
        Train the network
        '''
        self.step_4_1()
        self.step_4_2()
        pass
    def step_4_1(self):
        '''
        Prepare the training data and model for training
        '''
        self.prepare()
    def step_4_2(self):
        '''
        Start Training
        '''
        self.train_model()
        pass

    def step_5(self):
        '''
        Evaluate your model
        '''
        self.step_5_1()
        self.step_5_2()
        pass
    def step_5_1(self):
        '''
        Inspection of the loss function
        '''
        pass
    def step_5_2(self):
        '''
        Error mapping and quality metrics estimation
        '''
        self.quality()

    def step_6(self):
        '''
        Using the trained model
        '''
        self.step_6_1()
        self.step_6_2()
    def step_6_1(self):
        '''
        Generate prediction(s) from unseen dataset
        '''
        self.predict()
    def step_6_2(self):
        '''
        Assess predicted output
        '''
        self.assess()


    def model_specifics(self):
        pass

    def import_checks(self):
        pass

    # def __iter__(self):
    #     return iter(self.__dict__)

    # def __len__(self):
    #     return len(self.__dict__)

    # def __getitem__(self, arg):
    #     # return getattr(self,arg) #Move away from bloody dict
    #     return getattr(self, arg)

    # def __setitem__(self, key, value):
    #     setattr(self, key, value)
    #     # return

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
        return self.ref_str

    # def __repr__(self):
    #     self.dl4mic_model_config

    def append_config(self, config_dict):
        self.__dict__.update(config_dict)
        # return self.dl4mic_model_config

    def get_config(self):
        return self.__dict__

    def get_config_df(self):
        return pd.DataFrame(self.__dict__)

    # def data_checks(self):
    #     self.patch_size = checks.check_image_dims(
    #         self.patch_size, self.Training_source
    #     )

    def get_h5_path(self):
        self.h5_file_path = utils.get_h5_path(
            self.pretrained_model_path, self.Weights_choice
        )
        return self.h5_file_path

    def use_pretrained_model(self):
        pass
    def train_model():
        pass
    def get_model_params(self):
        return self[self.model_params]

    def get_config_params(self):
        return self[self.model_config]

    def model_export_tf(self, model, X_val):
        patch_size = self.batch_size
        model.export_TF(
            name=self.model_name,
            description=self.model_description,
            authors=self.authors,
            test_img=X_val[0, ..., 0],
            axes="YX",
            patch_shape=(
                self.patch_size,
                self.patch_size,
            ),
        )

    def data_checks(self, show_image=False):
        # checks.check_for_prexisiting_model()

        # image = checks.get_random_image(self.)
        # Training_source = self.Training_source
        Training_source = self.folders.Training_source
        output_folder = self.folders.output_folder
        patch_size = self.patch_size

        # checks.check_data(image)

        # filename = os.path.join(self.output_folder, "TrainingDataExample.png")
        # if show_image:
        # checks.display_image(image, filename)

        # checks.check_image_dims(image, self.patch_size)

        return checks.full(Training_source, output_folder, patch_size, show_image)

    def data_augmentation(self):
        pass

    def load_pretrained_model(self):
        if self.Use_pretrained_model:

            self.h5_file_path = utils.download_model(
                self.pretrained_model_path,
                self.pretrained_model_choice,
                self.pretrained_model_name,
                self.Weights_choice,
                self.model_path,
            )

            learning_rates_dict = utils.load_model(
                self.h5_file_path,
                self.pretrained_model_path,
                self.Weights_choice,
                self.initial_learning_rate,
            )

            self.append_config(learning_rates_dict)
            return self.h5_file_path
        else:
            pass
    def prepare(self):
        pass

    def train(self):
        pass  

    def augment(self):
        pass

    def checks(self):
        pass
    
    def reporting(self):
        pass

    def report(self, time_start=None, trained=None, show_image=False):
        # report_args = [
        #     "model_name",
        #     "model_path",
        #     "ref_str",
        #     "ref_aug",
        #     "Notebook_version",
        #     "initial_learning_rate",
        #     "number_of_steps",
        #     "percentage_validation",
        #     "image_patches",
        #     "loss_function",
        #     "batch_size",
        #     "patch_size",
        #     "Training_source",
        #     "number_of_epochs",
        #     # "time_start",
        #     "Use_Default_Advanced_Parameters",
        #     # "trained",
        #     "augmentation",
        #     "Use_pretrained_model",
        # ]
        # extra_args = {
        #     "time_start": time_start,
        #     "example_image": self.data.example_image,
        #     "trained": trained,
        # }

        # report_config = {key: self[key] for key in report_args}
        # report_config.update(extra_args)

        # # return reporting.pdf_export(**report_config)
        self.data.trained = trained
        self.data.time_start = time_start

        return reporting.pdf_export(
            self.model_name,
            self.model_path,
            self.ref_str,
            self.ref_aug,
            self.Notebook_version,
            self.initial_learning_rate,
            self.number_of_steps,
            self.percentage_validation,
            self.image_patches,
            self.loss_function,
            self.batch_size,
            self.patch_size,
            self.Training_source,
            self.number_of_epochs,
            self.Use_Default_Advanced_Parameters,
            self.data.time_start,
            self.data.example_image,
            self.data.trained,
            self.augmentation,
            self.Use_pretrained_model,
        )

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
        # return self.report(time_start=time_start, trained=None, show_image=False)
        return self.report(time_start=time_start, trained=trained, show_image=False)

    def post_report(
        self, X_train=None, X_test=None, time_start=None, trained=True, show_image=False
    ):
        return self.report(
            time_start=time_start, trained=trained, show_image=show_image
        )

    # def quality_stock(self):
    #     # Path(self.QC_model_path).mkdir(parents=True, exist_ok=True)

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

    def quality(self, history=None, show_images=False):

        # model_path = self.model_path
        # model_name = self.model_name

        # if self.QC_model_name is None:
        #     self.QC_model_name = model_name

        # if self.QC_model_path is None:
        #     self.QC_model_path = model_path

        # QC_model_name = self.QC_model_name
        # QC_model_path = self.QC_model_path

        # ref_str = self.ref_str
        # network = self.network
        # Use_the_current_trained_model = self.Use_the_current_trained_model
        # Source_QC_folder = self.Source_QC_folder
        # Target_QC_folder = self.Target_QC_folder
        # self.QC_dir = Path(os.path.join(QC_model_path,QC_model_name))
        # self.QC_dir.mkdir(parents=True, exist_ok=True)

        # return self.quality_stock()
        # def quality(self):

        if history != None:
            self.quality_extra(history=history)

        return quality.full(
            self.model_path,
            self.model_name,
            self.QC_model_name,
            self.QC_model_path,
            self.ref_str,
            self.network,
            self.Use_the_current_trained_model,
            self.Source_QC_folder,
            self.Target_QC_folder,
            show_images=show_images,
        )

    def predict(self):

        Prediction_model_path = self.folders.Prediction_model_path
        Prediction_model_name = self.Prediction_model_name

        return predict.full(Prediction_model_path, Prediction_model_name)

    def assess(self):

        Prediction_model_path = self.Prediction_model_path
        Prediction_model_name = self.Prediction_model_name
        Data_type = self.Data_type

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
            X_train=self.X_train,
            X_test=self.X_test,
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
        threshold = self.threshold
        X = Xdata[threshold:]
        X_val = Xdata[:threshold]
        self.X_train = X
        self.X_test = X_val
        return X, X_val

    # def default_augment(self):
    #     Use_Default_Augmentation_Parameters = self.Use_Default_Augmentation_Parameters

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
    # model_path = self.model_path
    # model_name = self.model_name

    # QC_model_name = self.QC_model_name
    # QC_model_path = self.QC_model_path

    # Source_QC_folder = self.Source_QC_folder
    # Target_QC_folder = self.Target_QC_folder
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
        patch_size = self.patch_size
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
    #     if self.Use_the_current_trained_model:
    #         self.QC_model_path = self.dl4mic_model_config[
    #             "model_path"
    #         ]
    #         self.QC_model_name = self.dl4mic_model_config[
    #             "model_name"
    #         ]
    #     # model = self."

    #     model_path = self.model_path
    #     model_name = self.model_name
    #     QC_model_name = self.QC_model_name
    #     QC_model_path = self.QC_model_path

    #     qc_folder = os.path.join(model_path, model_name, "Quality Control")

    #     quality.quality_tf(
    #         history, model_path, model_name, QC_model_name, QC_model_path
    #     )

    # return self.quality_stock()


"""
TODO
Fix loading of modules, unsure if the load when the 
class is loaded or if the init needs to happen first?
"""

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
#         self.model_config = ["train_steps_per_epoch","train_epochs","train_batch_size
#     def set_model_params(self):
#         self.model_params = ["model_name","model_path

#     # def __init__():
#     # datagen = N2V_DataGenerator()
#     # return
#     def get_ref(self):
#         return self.ref_str

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
#         self.patch_size = checks.check_image_dims(
#             self.patch_size, self.Training_source
#         )

#     def get_h5_path(self):
#         self.h5_file_path = os.path.join(
#             self.pretrained_model_path,
#             "weights_" + self.Weights_choice + ".h5",
#         )

#     def use_pretrained_model(self):
#         pass

#     def interface(self):
#         self.full_config = self.dl4mic_model_config
#         interface_dict = {
#             "name":self.model_name,
#             "basedir": self.model_path,
#             "train_steps_per_epoch":self.number_of_steps,
#             "train_epochs":self.number_of_epochs,
#             "train_batch_size":self.batch_size,
#         }
#         self.N2V_config.update(interface_dict)

#     def get_model_params(self):
#         return self.full_config[self.model_params]

#     def get_config_params(self):
#         return self.full_config[self.model_config]
#         # self.N2V_config["name = self.model_name
#         # self.N2V_config["basedir = self.model_path
#         # self.N2V_config["basedir = self.model_path
#     def model_export_tf(self,model,X_val):
#         patch_size = self.batch_size
#         model.export_TF(
#                 name=self.model_name,
#                 description=self.model_description,
#                 authors=self.authors,
#                 test_img=X_val[0,...,0], axes='YX',
#                 patch_shape=(self.patch_size,
#                              self.patch_size))
#     def model_specifics(self):
#         self.model_name = "N2V"
#         self.description = "Noise2Void 2D trained using ZeroCostDL4Mic.'"
#         self.authors = ["You
