# %%

# import pytest
import dl4mic.models as models
import dl4mic.utils as utils

model_config = {
    "model": None,
    "X_train": None,
    "X_test": None,
    "model_name": None,
    # "model_path": None,
    # "ref_str"=,
    "Notebook_version": 1.12,
    "initial_learning_rate": 0.0004,
    "number_of_steps": 100,
    "percentage_validation": 10,
    # "image_patches"=,
    "loss_function": "mse",
    "batch_size": 128,
    "patch_size": 64,
    "Training_source": None,
    "number_of_epochs": 100,
    "Use_Default_Advanced_Parameters": False,
    "Use_Data_augmentation": False,
    "trained": False,
    "augmentation": False,
    "pretrained_model": False,
    "pretrained_model_choice": "Model_from_file",
    "percentage_validation": 10,
    "Use_pretrained_model": False,
}


def test_dl4mic_model():
    dl4mic_model = models.DL4MicModel()


def test_N2V():
    import os

    os.environ["KERAS_BACKEND"] = "tensorflow"

    from n2v.models import N2VConfig, N2V
    from csbdeep.utils import plot_history
    from n2v.utils.n2v_utils import manipulate_val_data
    from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
    from csbdeep.io import save_tiff_imagej_compatible

    dl4mic_model = models.N2V(model_config)
    dl4mic_model.append_config({"Training_source": "Training"})
    print(dl4mic_model["Training_source"])

    Training_source = dl4mic_model["Training_source"]
    print(Training_source)
    datagen = N2V_DataGenerator()
    training_images = Training_source
    imgs = datagen.load_imgs_from_directory(directory=Training_source)

    example_image = dl4mic_model.data_checks()
    dl4mic_model.data_augmentation()
    h5_file_path = dl4mic_model.load_pretrained_model()

    Xdata = datagen.generate_patches_from_list(
        imgs,
        shape=(dl4mic_model["patch_size"], dl4mic_model["patch_size"]),
        augment=dl4mic_model["Use_Data_augmentation"],
    )

    dl4mic_model.gleen_data(Xdata)

    shape_of_Xdata = Xdata.shape

    threshold = dl4mic_model["threshold"]
    image_patches = dl4mic_model["image_patches"]

    X = Xdata[threshold:]
    X_val = Xdata[:threshold]

    print(shape_of_Xdata[0], "patches created.")
    print(
        dl4mic_model["threshold"],
        "patch images for validation (",
        dl4mic_model["percentage_validation"],
        "%).",
    )
    print(image_patches - threshold, "patch images for training.")

    config = N2VConfig(
        X,
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
        basedir=dl4mic_model["model_path"],
    )
    if dl4mic_model["Use_pretrained_model"]:
        model.load("h5_file_path")

    print("Setup done.")
    print(config)
    dl4mic_model.check_model_params()
    pdf = dl4mic_model.report(X_train=X, X_test=X_val, show_image=True)


test_N2V()
# %%
