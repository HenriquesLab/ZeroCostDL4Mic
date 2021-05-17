from n2v.models import N2VConfig, N2V
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from csbdeep.io import save_tiff_imagej_compatible
from pathlib import Path
import os
from tifffile import imread, imsave
from .. import predict, quality, checks, utils, prepare, reporting, assess
import time
from skimage import img_as_float32
import numpy as np

from dl4mic import models

# def __init__(self):
#     return self.N2V

class N2V(models.DL4MicModelTF):
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
            self.dl4mic_model_config["number_of_steps"] = self.get_default_steps(
                self.dl4mic_model_config["shape_of_Xdata"]
            )

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
        imgs = datagen.load_imgs_from_directory(
            directory=dl4mic_model["Training_source"]
        )

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
