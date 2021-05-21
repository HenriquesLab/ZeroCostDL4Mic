from . import bcolors
import os


def full(Prediction_model_path, Prediction_model_name):
    if (Prediction_model_path or Prediction_model_name) is not None:
        check_folder(Prediction_model_path, Prediction_model_name)
    pass


def check_folder(Prediction_model_path, Prediction_model_name):

    # full_Prediction_model_path = (
    #     Prediction_model_path + "/" + Prediction_model_name + "/"
    # )
    try:
        full_Prediction_model_path = os.path.join(
            Prediction_model_path, Prediction_model_name
        )
    except TypeError:
        print("Bad or empty model path or name")
        return
    if os.path.exists(full_Prediction_model_path):
        print("The " + Prediction_model_name + " network will be used.")
        return
    else:
        print(bcolors.WARNING + "!! WARNING: The chosen model does not exist !!")
        print(
            "Please make sure you provide a valid model path and model name before proceeding further."
        )
    return
