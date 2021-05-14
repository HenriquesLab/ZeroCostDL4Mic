from . import bcolors
import time
import pandas as pd
import os
import wget
import shutil

import inspect
import functools


def info_about_model(Use_pretrained_model, h5_file_path):
    # Display info about the pretrained model to be loaded (or not)
    if Use_pretrained_model:
        print("Weights found in:")
        print(h5_file_path)
        print("will be loaded prior to training.")
    else:
        print(bcolors.WARNING + "No pretrained network will be used.")


def time_elapsed(time_start):
    dt = time.time() - time_start
    mins, sec = divmod(dt, 60)
    hour, mins = divmod(mins, 60)
    print("Time elapsed:", hour, "hour(s)", mins, "min(s)", round(sec), "sec(s)")
    return hour, mins, sec


def read_latest_notebook_version(Notebook_version, csv_url):
    Latest_notebook_version = pd.read_csv(csv_url)
    #  "https://raw.githubusercontent.com/HenriquesLab/ZeroCostDL4Mic/master/Colab_notebooks/Latest_ZeroCostDL4Mic_Release.csv"
    print("Notebook version: " + Notebook_version[0])
    strlist = Notebook_version[0].split(".")
    Notebook_version_main = strlist[0] + "." + strlist[1]
    if Notebook_version_main == Latest_notebook_version.columns:
        print("This notebook is up-to-date.")
    else:
        print(
            bcolors.WARNING
            + "A new version of this notebook has been released. We recommend that you download it at https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki"
        )
    return Latest_notebook_version


def get_h5_path(pretrained_model_path, Weights_choice):
    h5_file_path = os.path.join(
        pretrained_model_path,
        "weights_" + Weights_choice + ".h5",
    )
    return h5_file_path


def download_model(
    pretrained_model_path,
    pretrained_model_choice,
    pretrained_model_name,
    Weights_choice,
    output_folder,
):
    # params.Pretrained_model_choice.Model_from_file

    if pretrained_model_choice == "Model_from_file":
        h5_file_path = os.path.join(
            pretrained_model_path, "weights_" + str(Weights_choice) + ".h5"
        )
    if pretrained_model_choice == "Model_name":
        # pretrained_model_name = "Model_name"
        pretrained_model_path = os.path.join(output_folder, pretrained_model_name)
        print("Downloading the model")
        if os.path.exists(pretrained_model_path):
            shutil.rmtree(pretrained_model_path)
        os.makedirs(pretrained_model_path)
        wget.download("", pretrained_model_path)
        wget.download("", pretrained_model_path)
        wget.download("", pretrained_model_path)
        wget.download("", pretrained_model_path)
        h5_file_path = os.path.join(
            pretrained_model_path, "weights_" + Weights_choice + ".h5"
        )
    return h5_file_path


def load_model(
    h5_file_path, pretrained_model_path, Weights_choice, initial_learning_rate
):
    # If the model path contains a pretrain model, we load the training rate,
    if os.path.exists(h5_file_path):
        # Here we check if the learning rate can be loaded from the quality control folder
        if os.path.exists(
            os.path.join(
                pretrained_model_path, "Quality Control", "training_evaluation.csv"
            )
        ):

            with open(
                os.path.join(
                    pretrained_model_path, "Quality Control", "training_evaluation.csv"
                ),
                "r",
            ) as csvfile:
                csvRead = pd.read_csv(csvfile, sep=",")
                # print(csvRead)

                if (
                    "learning rate" in csvRead.columns
                ):  # Here we check that the learning rate column exist (compatibility with model trained un ZeroCostDL4Mic bellow 1.4)
                    print("pretrained network learning rate found")
                    # find the last learning rate
                    lastLearningRate = csvRead["learning rate"].iloc[-1]
                    # Find the learning rate corresponding to the lowest validation loss
                    min_val_loss = csvRead[
                        csvRead["val_loss"] == min(csvRead["val_loss"])
                    ]
                    # print(min_val_loss)
                    bestLearningRate = min_val_loss["learning rate"].iloc[-1]

                    if Weights_choice == "last":
                        print("Last learning rate: " + str(lastLearningRate))

                    if Weights_choice == "best":
                        print(
                            "Learning rate of best validation loss: "
                            + str(bestLearningRate)
                        )

                    if (
                        not "learning rate" in csvRead.columns
                    ):  # if the column does not exist, then initial learning rate is used instead
                        bestLearningRate = initial_learning_rate
                        lastLearningRate = initial_learning_rate
                    print(
                        bcolors.WARNING
                        + "WARNING: The learning rate cannot be identified from the pretrained network. Default learning rate of "
                        + str(bestLearningRate)
                        + " will be used instead"
                    )

        # Compatibility with models trained outside ZeroCostDL4Mic but default learning rate will be used
        if not os.path.exists(
            os.path.join(
                pretrained_model_path, "Quality Control", "training_evaluation.csv"
            )
        ):
            print(
                bcolors.WARNING
                + "WARNING: The learning rate cannot be identified from the pretrained network. Default learning rate of "
                + str(initial_learning_rate)
                + " will be used instead"
            )
            bestLearningRate = initial_learning_rate
            lastLearningRate = initial_learning_rate
        return {"bestLearningRate": bestLearningRate, "lastLearningRate": lastLearningRate}
    return {"bestLearningRate": initial_learning_rate, "lastLearningRate": initial_learning_rate}
    

def dl4mic(f):
    """Make function ignore unmatched kwargs.

    If the function already has the catch all **kwargs, do nothing.
    """
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in inspect.signature(f).parameters.values()
    ):
        return f
    #
    @functools.wraps(f)
    def inner(*args, **kwargs):
        # For each keyword arguments recognised by f,
        # take their binding from **kwargs received
        filtered_kwargs = {
            name: kwargs[name]
            for name, param in inspect.signature(f).parameters.items()
            if (
                param.kind is inspect.Parameter.KEYWORD_ONLY
                or param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            )
            and name in kwargs
        }
        return f(*args, **filtered_kwargs)

    return inner
