# %%

# import pytest
import dl4mic.models as models
import dl4mic.utils as utils
import time
import pandas as pd
import pytest

model_configs = [
    {
        "model": None,
        "X_train": None,
        "X_test": None,
        # "model_name": None,
        # "model_path": None,
        # "ref_str"=,
        "Notebook_version": 1.12,
        "initial_learning_rate": 0.0004,
        "number_of_steps": 1,
        "percentage_validation": 10,
        # "image_patches"=,
        "loss_function": "mse",
        "batch_size": 128,
        "patch_size": 64,
        "Training_source": "tests/n2v/Training",
        "pretrained_model_path": "tests/n2v/weights_last.h5",
        "pretrained_model_name": "Model_name",
        "number_of_epochs": 1,
        "Use_Default_Advanced_Parameters": False,
        "Use_Data_augmentation": False,
        # "trained": False,
        # "augmentation": False,
        "pretrained_model": False,
        "pretrained_model_choice": "Model_from_file",
        "percentage_validation": 10,
        "Use_pretrained_model": True,
        "Use_the_current_trained_model": True,
        "Source_QC_folder": None,
        "Target_QC_folder": None,
        "Prediction_model_folder": None,
        "QC_model_name": "n2v",
        "Data_folder": None,
        "Data_type": models.params.Data_type.SINGLE_IMAGES,
        "Prediction_model_name": None,
        "Prediction_model_path": None,
    }
    ,
    {
        "model": None,
        "X_train": None,
        "X_test": None,
        # "model_name": None,
        # "model_path": None,
        # "ref_str"=,
        "Notebook_version": 1.12,
        "initial_learning_rate": 0.0004,
        "number_of_steps": 1,
        "percentage_validation": 10,
        # "image_patches"=,
        "loss_function": "mse",
        "batch_size": 128,
        "patch_size": 64,
        "Training_source": "tests/n2v/Training",
        "pretrained_model_path": "tests/n2v/weights_last.h5",
        "pretrained_model_name": "Model_name",
        "number_of_epochs": 1,
        "Use_Default_Advanced_Parameters": True,
        "number_of_steps": 100,
        "Use_Data_augmentation": False,
        # "trained": False,
        # "augmentation": False,
        "pretrained_model": False,
        "pretrained_model_choice": "Model_from_file",
        "percentage_validation": 10,
        "Use_pretrained_model": True,
        "Use_the_current_trained_model": True,
        "Source_QC_folder": None,
        "Target_QC_folder": None,
        "Prediction_model_folder": None,
        "QC_model_name": "n2v",
        "Data_folder": None,
        "Data_type": models.params.Data_type.SINGLE_IMAGES,
        "Prediction_model_name": None,
        "Prediction_model_path": None,
    }
]

# Use_Default_Advanced_Parameters = [True,False]
# %%
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

    model_config = model_configs[0]
    dl4mic_model = models.N2V(model_config)
    # dl4mic_model.append_config({"Training_source": "Training"})
    # print(dl4mic_model["Training_source"])

    # Training_source = dl4mic_model["Training_source"]
    # print(Training_source)
    datagen = N2V_DataGenerator()
    # training_images = Training_source
    imgs = datagen.load_imgs_from_directory(directory=dl4mic_model["Training_source"])

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
        basedir="tests",
    )
    if dl4mic_model["Use_pretrained_model"]:
        model.load_weights("weights_last.h5")

    print("Setup done.")
    print(config)
    dl4mic_model.check_model_params()
    pdf = dl4mic_model.pre_report(X_train=X, X_test=X_val, show_image=False)

    # def test_check_quality():

    # start = time.time()

    dl4mic_model["start"] = time.time()

    # TF1 Hack
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    tf.__version__ = 1.14

    # model.load_weights("n2v/weights_last.h5")
    # history = model.train(X, X_val)
    history = [0, 1, 2]
    print("Training done.")
    # lossData_df = pd.DataFrame(history.history)
    # dl4mic_model.save_model(model)
    dl4mic_model.quality(history)
    # dl4mic_model.quality()
    pdf = dl4mic_model.post_report(show_image=False)
    dl4mic_model.predict()
    dl4mic_model.assess()


@pytest.mark.parametrize("model_config", model_configs)
def test_N2V_short(model_config):
    import os

    os.environ["KERAS_BACKEND"] = "tensorflow"

    from n2v.internals.N2V_DataGenerator import N2V_DataGenerator

    dl4mic_model = models.N2V(model_config)
    datagen = N2V_DataGenerator()
    imgs = datagen.load_imgs_from_directory(directory=dl4mic_model["Training_source"])

    Xdata = datagen.generate_patches_from_list(
        imgs,
        shape=(dl4mic_model["patch_size"], dl4mic_model["patch_size"]),
        augment=dl4mic_model["Use_Data_augmentation"],
    )

    dl4mic_model.pre_training(Xdata)

    dl4mic_model["start"] = time.time()

    # TF1 Hack
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    tf.__version__ = 1.14

    model = dl4mic_model.get_model()
    threshold = dl4mic_model["threshold"]

    X = Xdata[threshold:]
    X_val = Xdata[:threshold]

    history = model.train(X, X_val)
    print("Training done.")

    pdf_post = dl4mic_model.post_report(history)


# def test_N2V_short(model_config):
@pytest.mark.parametrize("model_config", model_configs)
def test_N2V_very_short(model_config):
    models.N2V(model_config).run()

model_config_care = {
        "model": None,
        "X_train": None,
        "X_test": None,
        # "model_name": None,
        # "model_path": None,
        # "ref_str"=,
        "Notebook_version": 1.12,
        "initial_learning_rate": 0.0004,
        "number_of_steps": 1,
        "percentage_validation": 10,
        # "image_patches"=,
        "loss_function": "mse",
        "batch_size": 128,
        "patch_size": 64,
        "Training_source": "tests/n2v/Training",
        "pretrained_model_path": "tests/n2v/weights_last.h5",
        "pretrained_model_name": "Model_name",
        "number_of_epochs": 1,
        "Use_Default_Advanced_Parameters": False,
        "Use_Data_augmentation": False,
        # "trained": False,
        # "augmentation": False,
        "pretrained_model": False,
        "pretrained_model_choice": models.params.Pretrained_model_choice.MODEL_FROM_FILE,
        "percentage_validation": 10,
        "Use_pretrained_model": True,
        "Use_the_current_trained_model": True,
        "Source_QC_folder": None,
        "Target_QC_folder": None,
        "Prediction_model_folder": None,
        "QC_model_name": "n2v",
        "Data_folder": None,
        "Data_type": models.params.Data_type.SINGLE_IMAGES,
        "Prediction_model_name": None,
        "Prediction_model_path": None,
    }
@pytest.mark.parametrize("model_config_care", [model_config_care])
def test_CARE_very_short(model_config_care):
    models.CARE(model_config_care).run()



# def n2v_get_model(dl4mic_model, Xdata):

#     ################ N2V ######################

#     from n2v.models import N2VConfig, N2V
#     from csbdeep.utils import plot_history
#     from n2v.utils.n2v_utils import manipulate_val_data
#     from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
#     from csbdeep.io import save_tiff_imagej_compatible

#     threshold = dl4mic_model["threshold"]
#     image_patches = dl4mic_model["image_patches"]
#     shape_of_Xdata = dl4mic_model["shape_of_Xdata"]

#     print(shape_of_Xdata[0], "patches created.")
#     print(
#         dl4mic_model["threshold"],
#         "patch images for validation (",
#         dl4mic_model["percentage_validation"],
#         "%).",
#     )
#     print(image_patches - threshold, "patch images for training.")

#     config = N2VConfig(
#         dl4mic_model["X_train"],
#         unet_kern_size=3,
#         train_steps_per_epoch=dl4mic_model["number_of_steps"],
#         train_epochs=dl4mic_model["number_of_epochs"],
#         train_loss=dl4mic_model["loss_function"],
#         batch_norm=True,
#         train_batch_size=dl4mic_model["batch_size"],
#         n2v_perc_pix=0.198,
#         n2v_manipulator="uniform_withCP",
#         n2v_neighborhood_radius=5,
#         train_learning_rate=dl4mic_model["initial_learning_rate"],
#     )

#     model = N2V(
#         config=config,
#         name=dl4mic_model["model_name"],
#         basedir="tests",
#     )

#     print("Setup done.")
#     print(config)
#     return model

#     # if dl4mic_model["Use_pretrained_model"]:
#     #     model.load_weights("weights_last.h5")

#     ###############################################


# # test_N2V()
# # %%
