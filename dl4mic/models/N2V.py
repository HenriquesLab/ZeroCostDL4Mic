from pathlib import Path
import os
from tifffile import imread, imsave
from tifffile.tifffile import read_uic1tag
from .. import predict, quality, checks, utils, prepare, reporting, assess
import time
from skimage import img_as_float32
import numpy as np
from csbdeep.io import save_tiff_imagej_compatible

from n2v.models import N2VConfig, N2V
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from csbdeep.io import save_tiff_imagej_compatible

from .. import models

from typing import List

# def __init__(self):
#     return self.N2V
# defaults = {
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
#             "Pretrained_model_choice": models.params.Pretrained_model_choice.MODEL_NAME,
#             "Weights_choice": models.params.Pretrained_model_choice.BEST,
#          }


class N2V(models.DL4MicModelTF):
    model_name: str = None
    model_path: str = None
    ref_str = '- Noise2Void: Krull, Alexander, Tim-Oliver Buchholz, and Florian Jug. "Noise2void-learning denoising from single noisy images." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.'
    initial_learning_rate: float = 0.0004
    number_of_steps: int = 100
    percentage_validation: int = 10
    # image_patches=
    loss_function: str = "mse"
    batch_size: int = 128
    patch_size: int = 64
    # Training_source: None
    number_of_epochs: int = 100
    Use_Default_Advanced_Parameters: bool = False
    trained: bool = False
    augmentation: bool = False
    pretrained_model: bool = False
    Pretrained_model_choice: str = models.params.Pretrained_model_choice.MODEL_NAME
    Weights_choice: str = models.params.Weights_choice.BEST
    network: str = "Noise2Void"
    model_name: str = "n2v"
    description: str = "Noise2Void 2D trained using ZeroCostDL4Mic.'"
    authors: List[str] = ["You"]
    # import N2V
    # config=None
    # super().__init__(**model_config)
    # self.dl4mic_model_config={}
    # def init(self):
    # self.network = "Noise2Void"
    # self.model_name = "n2v"
    # self.description = "Noise2Void 2D trained using ZeroCostDL4Mic.'"
    # self.authors = ["You"]
    # self.ref_str = '- Noise2Void: Krull, Alexander, Tim-Oliver Buchholz, and Florian Jug. "Noise2void-learning denoising from single noisy images." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.'
    # pass

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
            "name": self.model_name,
            "basedir": self.model_path,
            "train_steps_per_epoch": self.number_of_steps,
            "train_epochs": self.number_of_epochs,
            "train_batch_size": self.batch_size,
            "directory": self.Training_source,
        }
        self.append_config(interface_dict)

    def model_specifics(self):
        pass

    def gleen_data(self, Xdata):
        self.shape_of_Xdata = Xdata.shape

        # self.shape_of_Xdata = shape_of_Xdata

        self.get_threshold(self.shape_of_Xdata)
        self.get_image_patches(self.shape_of_Xdata)
        if self.Use_Default_Advanced_Parameters:
            self.number_of_steps = self.get_default_steps(self.shape_of_Xdata)

    def get_threshold(self, shape_of_Xdata):
        self.threshold = int(shape_of_Xdata[0] * (self.percentage_validation / 100))
        return self.threshold

    def get_image_patches(self, shape_of_Xdata):
        self.image_patches = int(shape_of_Xdata[0])
        return self.image_patches

    def get_default_steps(self, shape_of_Xdata):
        self.number_of_steps = int(shape_of_Xdata[0] / self.batch_size) + 1
        return self.number_of_steps

    def save_model(model):
        pass

    def quality_extra(self, history=None):
        # history = self.history
        # model_path = self.model_path
        # model_name = self.model_name
        # QC_model_name = self.QC_model_name
        # QC_model_path = self.QC_model_path

        if self.data.history is not None:
            history = self.data.history
        if history is None:
            return

        quality.quality_tf(
            history,
            self.model_path,
            self.model_name,
            self.QC_model_name,
            self.QC_model_path,
        )

    def get_model(self):
        return get_model(
            self.threshold,
            self.image_patches,
            self.shape_of_Xdata,
            self.X_train,
            self.percentage_validation,
            self.number_of_steps,
            self.number_of_epochs,
            self.initial_learning_rate,
            self.loss_function,
            self.batch_size,
            self.model_name,
        )

    def run(self):
        # import os
        # TF1 Hack
        import tensorflow.compat.v1 as tf

        tf.disable_v2_behavior()
        tf.__version__ = 1.14
        os.environ["KERAS_BACKEND"] = "tensorflow"

        # from n2v.internals.N2V_DataGenerator import N2V_DataGenerator

        # dl4mic_model = self.dl4mic_model_config

        # datagen = N2V_DataGenerator()

        imgs = get_imgs(
            self.Training_source, self.patch_size, self.Use_Data_augmentation
        )

        Xdata = get_Xdata(imgs, self.patch_size, self.Use_Data_augmentation)

        self.pre_training(Xdata)

        self.start = time.time()

        model = self.get_model()
        # threshold = self.threshold

        X = Xdata[self.threshold :]
        X_val = Xdata[: self.threshold]

        self.data.X_train = X
        self.data.X_test = X_val

        self.data.history = model.train(X, X_val)
        print("Training done.")

        pdf_post = self.post_report(self.data.history)
        return self


def predict_on_folder(
    Prediction_model_name, Prediction_model_path, Data_folder, Result_folder, Data_type
):

    # Activate the pretrained model.
    config = None
    model = N2V(config, Prediction_model_name, basedir=Prediction_model_path)

    thisdir = Path(Data_folder)
    outputdir = Path(Result_folder)

    # r=root, d=directories, f = files
    for r, d, f in os.walk(thisdir):
        for file in f:
            if ".tif" in file:
                print(os.path.join(r, file))

    if Data_type == models.params.Data_type.SINGLE_IMAGES:
        print("Single images are now beeing predicted")

    # Loop through the files
    for r, d, f in os.walk(thisdir):
        for file in f:
            base_filename = os.path.basename(file)
            input_train = imread(os.path.join(r, file))
            pred_train = model.predict(input_train, axes="YX", n_tiles=(2, 1))
            save_tiff_imagej_compatible(
                os.path.join(outputdir, base_filename), pred_train, axes="YX"
            )

    print("Images saved into folder:", Result_folder)

    if Data_type == models.params.Data_type.STACKS:
        print("Stacks are now beeing predicted")
        for r, d, f in os.walk(thisdir):
            for file in f:
                base_filename = os.path.basename(file)
                timelapse = imread(os.path.join(r, file))
                n_timepoint = timelapse.shape[0]
                prediction_stack = np.zeros(
                    (n_timepoint, timelapse.shape[1], timelapse.shape[2])
                )

            for t in range(n_timepoint):
                img_t = timelapse[t]
                prediction_stack[t] = model.predict(img_t, axes="YX", n_tiles=(2, 1))

            prediction_stack_32 = img_as_float32(prediction_stack, force_copy=False)
            imsave(os.path.join(outputdir, base_filename), prediction_stack_32)


def get_model(
    threshold,
    image_patches,
    shape_of_Xdata,
    X_train,
    percentage_validation,
    number_of_steps,
    number_of_epochs,
    initial_learning_rate,
    loss_function,
    batch_size,
    model_name,
):

    # dl4mic_model = self.dl4mic_model_config
    # def n2v_get_model(dl4mic_model, Xdata):

    ################ N2V ######################

    from n2v.models import N2VConfig, N2V
    from csbdeep.utils import plot_history
    from n2v.utils.n2v_utils import manipulate_val_data
    from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
    from csbdeep.io import save_tiff_imagej_compatible

    # threshold = self.threshold
    # image_patches = self.image_patches
    # shape_of_Xdata = self.shape_of_Xdata

    print(shape_of_Xdata[0], "patches created.")
    print(
        threshold,
        "patch images for validation (",
        percentage_validation,
        "%).",
    )
    print(image_patches - threshold, "patch images for training.")

    config = N2VConfig(
        X_train,
        unet_kern_size=3,
        train_steps_per_epoch=number_of_steps,
        train_epochs=number_of_epochs,
        train_loss=loss_function,
        batch_norm=True,
        train_batch_size=batch_size,
        n2v_perc_pix=0.198,
        n2v_manipulator="uniform_withCP",
        n2v_neighborhood_radius=5,
        train_learning_rate=initial_learning_rate,
    )

    model = N2V(
        config=config,
        name=model_name,
        basedir="tests",
    )

    print("Setup done.")
    print(config)
    return model


def get_Xdata(imgs, patch_size, Use_Data_augmentation):
    from n2v.internals.N2V_DataGenerator import N2V_DataGenerator

    datagen = N2V_DataGenerator()

    Xdata = datagen.generate_patches_from_list(
        imgs,
        shape=(patch_size, patch_size),
        augment=Use_Data_augmentation,
    )
    return Xdata


def get_imgs(Training_source, patch_size, Use_Data_augmentation):

    # dl4mic_model = self.dl4mic_model_config

    datagen = N2V_DataGenerator()
    imgs = datagen.load_imgs_from_directory(directory=Training_source)
    return imgs
