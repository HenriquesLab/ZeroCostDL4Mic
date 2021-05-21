from . import reporting
from glob import glob
# import io
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tifffile.tifffile import imread, imsave
from . import bcolors
import shutil
import os
from pathlib import Path
import csv
from skimage.metrics import structural_similarity
import numexpr
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import io


# qc_folder = "Quality Control"


def quality_folder_reset(QC_model_path, QC_model_name):
    folder = os.path.join(QC_model_path, QC_model_name)
    if os.path.exists(folder):
        shutil.rmtree(folder)

    Path(folder).mkdir(parents=True, exist_ok=True)
    return folder


def df_to_csv(df, QC_model_path, QC_model_name):
    # lossDataCSVpath = os.path.join(model_path+'/'+model_name+'/Quality Control/','training_evaluation.csv')
    try:
        lossDataCSVpath = os.path.join(
            QC_model_path, QC_model_name, "training_evaluation.csv"
        )
        df.to_csv(lossDataCSVpath)
        return lossDataCSVpath
    except FileNotFoundError:
        print("Couldn't find training_evaluation")
        return None


    # with open(lossDataCSVpath, 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['loss','val_loss', 'learning rate'])
    #     for i in range(len(history.history['loss'])):
    #         writer.writerow([history.history['loss'][i], history.history['val_loss'][i], history.history['lr'][i]])


def qc_model_checks(
    QC_model_name, QC_model_path, model_name, model_path, Use_the_current_trained_model
):
    # Here we define the loaded model name and path
    # QC_model_name = os.path.basename(QC_model_folder)
    # QC_model_path = os.path.dirname(QC_model_folder)

    if Use_the_current_trained_model:
        QC_model_name = model_name
        QC_model_path = model_path

    # full_QC_model_path = QC_model_path+'/'+QC_model_name+'/'
    full_QC_model_path = os.path.join(QC_model_path, QC_model_name)

    if os.path.exists(full_QC_model_path):
        print("The " + QC_model_name + " network will be evaluated")
    else:
        print(bcolors.WARNING + "!! WARNING: The chosen model does not exist !!")
        print(
            "Please make sure you provide a valid model path and model name before proceeding further."
        )
    return full_QC_model_path


def inspect_loss(QC_model_name, QC_model_path, show_images=False):
    return display_training_errors(QC_model_name, QC_model_path,show_images=show_images)


# def make_dir_at_file(file):

# plot of training errors vs. epoch number
def display_training_errors(QC_model_name, QC_model_path,show_images=False):
    # Pandas surely?
    lossDataFromCSV = []
    vallossDataFromCSV = []

    qd_training_eval_csv = os.path.join(
        QC_model_path, QC_model_name, "training_evaluation.csv"
    )

    Path(qd_training_eval_csv).parent.mkdir(parents=True, exist_ok=True)
    print(Path(qd_training_eval_csv).parent)
    try:
        with open(qd_training_eval_csv, "r") as csvfile:
            csvRead = csv.reader(csvfile, delimiter=",")
            next(csvRead)
            for row in csvRead:
                lossDataFromCSV.append(float(row[0]))
                vallossDataFromCSV.append(float(row[1]))

            epochNumber = range(len(lossDataFromCSV))
            plt.figure(figsize=(15, 10))

            plt.subplot(2, 1, 1)
            plt.plot(epochNumber, lossDataFromCSV, label="Training loss")
            plt.plot(epochNumber, vallossDataFromCSV, label="Validation loss")
            plt.title("Training loss and validation loss vs. epoch number (linear scale)")
            plt.ylabel("Loss")
            plt.xlabel("Epoch number")
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.semilogy(epochNumber, lossDataFromCSV, label="Training loss")
            plt.semilogy(epochNumber, vallossDataFromCSV, label="Validation loss")
            plt.title("Training loss and validation loss vs. epoch number (log scale)")
            plt.ylabel("Loss")
            plt.xlabel("Epoch number")
            plt.legend()
            loss_curve_path = os.path.join(
                QC_model_path, QC_model_name, "lossCurvePlots.png"
            )
            plt.savefig(loss_curve_path)
            if show_images:
                plt.show()
            else:
                plt.close()
    except FileNotFoundError:
        print("CSV not found")
    # Source_QC_folder = ""  # @param{type:"string"}
    # Target_QC_folder = ""  # @param{type:"string"}

    # # Create a quality control/Prediction Folder
    # if os.path.exists(
    #     QC_model_path + "/" + QC_model_name + "/Quality Control/Prediction"
    # ):
    #     shutil.rmtree(
    #         QC_model_path + "/" + QC_model_name + "/Quality Control/Prediction"
    #     )

    # os.makedirs(QC_model_path + "/" + QC_model_name + "/Quality Control/Prediction")

    # # tf_model_predictions_save(model,Source_QC_folder,QC_model_path,QC_model_name)

    # # Activate the pretrained model.


def tf_model_predictions_save(
    model_training, Source_QC_folder, QC_model_path, QC_model_name
):
    # model_training = N2V(config=None, name=QC_model_name, basedir=QC_model_path)

    qc_image_path = os.path.join(
        QC_model_path, QC_model_name, "Prediction"
    )

    # List Tif images in Source_QC_folder
    Source_QC_folder_tif = Source_QC_folder + "/*.tif"
    Z = sorted(glob(Source_QC_folder_tif))
    Z = list(map(imread, Z))

    print("Number of test dataset found in the folder: " + str(len(Z)))

    # Perform prediction on all datasets in the Source_QC folder
    for filename in os.listdir(Source_QC_folder):
        img = imread(os.path.join(Source_QC_folder, filename))
        predicted = model_training.predict(img, axes="YX", n_tiles=(2, 1))
        # os.chdir(qc_image_path) #Lethal surely
        imsave(filename, predicted)


def ssim(img1, img2):
    return structural_similarity(
        img1,
        img2,
        data_range=1.0,
        full=True,
        gaussian_weights=True,
        use_sample_covariance=False,
        sigma=1.5,
    )


def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """This function is adapted from Martin Weigert"""
    """Percentile-based image normalization."""

    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(
    x, mi, ma, clip=False, eps=1e-20, dtype=np.float32
):  # dtype=np.float32
    """This function is adapted from Martin Weigert"""
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x


def norm_minmse(gt, x, normalize_gt=True):
    """This function is adapted from Martin Weigert"""

    """
    normalizes and affinely scales an image pair such that the MSE is minimized  
    
    Parameters
    ----------
    gt: ndarray
        the ground truth image      
    x: ndarray
        the image that will be affinely scaled 
    normalize_gt: bool
        set to True of gt image should be normalized (default)
    Returns
    -------
    gt_scaled, x_scaled 
    """
    if normalize_gt:
        gt = normalize(gt, 0.1, 99.9, clip=False).astype(np.float32, copy=False)
    x = x.astype(np.float32, copy=False) - np.mean(x)
    # x = x - np.mean(x)
    gt = gt.astype(np.float32, copy=False) - np.mean(gt)
    # gt = gt - np.mean(gt)
    scale = np.cov(x.flatten(), gt.flatten())[0, 1] / np.var(x.flatten())
    return gt, scale * x


# Source_QC_folder = "" #@param{type:"string"}
# Target_QC_folder = "" #@param{type:"string"}

# # Create a quality control/Prediction Folder
# if os.path.exists(QC_model_path+"/"+QC_model_name+"/Quality Control/Prediction"):
#   shutil.rmtree(QC_model_path+"/"+QC_model_name+"/Quality Control/Prediction")

# os.makedirs(QC_model_path+"/"+QC_model_name+"/Quality Control/Prediction")

# # Activate the pretrained model.
# model_training = N2V(config=None, name=QC_model_name, basedir=QC_model_path)


# # List Tif images in Source_QC_folder
# Source_QC_folder_tif = Source_QC_folder+"/*.tif"
# Z = sorted(glob(Source_QC_folder_tif))
# Z = list(map(imread,Z))

# print('Number of test dataset found in the folder: '+str(len(Z)))


def create_qc_csv(QC_model_path, QC_model_name, Source_QC_folder, Target_QC_folder):

    # Open and create the csv file that will contain all the QC metrics

    qc_csv_path = os.path.join(
        QC_model_path,
        QC_model_name,
        "QC_metrics_"+QC_model_name+".csv",
    )
    with open(
        qc_csv_path,
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        # Write the header in the csv file
        writer.writerow(
            [
                "image #",
                "Prediction v. GT mSSIM",
                "Input v. GT mSSIM",
                "Prediction v. GT NRMSE",
                "Input v. GT NRMSE",
                "Prediction v. GT PSNR",
                "Input v. GT PSNR",
            ]
        )

        # Let's loop through the provided dataset in the QC folders
        try:
            for i in os.listdir(Source_QC_folder):
                if not os.path.isdir(os.path.join(Source_QC_folder, i)):
                    print("Running QC on: " + i)
                    # -------------------------------- Target test data (Ground truth) --------------------------------
                    test_GT = io.imread(os.path.join(Target_QC_folder, i))

                    # -------------------------------- Source test data --------------------------------
                    test_source = io.imread(os.path.join(Source_QC_folder, i))

                    # Normalize the images wrt each other by minimizing the MSE between GT and Source image
                    test_GT_norm, test_source_norm = norm_minmse(
                        test_GT, test_source, normalize_gt=True
                    )

                    # -------------------------------- Prediction --------------------------------
                    test_prediction = io.imread(
                        os.path.join(
                            QC_model_path,
                            QC_model_name,
                            "Prediction",
                            i,
                        )
                    )

                    # Normalize the images wrt each other by minimizing the MSE between GT and prediction
                    test_GT_norm, test_prediction_norm = norm_minmse(
                        test_GT, test_prediction, normalize_gt=True
                    )

                    # -------------------------------- Calculate the metric maps and save them --------------------------------

                    # Calculate the SSIM maps
                    index_SSIM_GTvsPrediction, img_SSIM_GTvsPrediction = ssim(
                        test_GT_norm, test_prediction_norm
                    )
                    index_SSIM_GTvsSource, img_SSIM_GTvsSource = ssim(
                        test_GT_norm, test_source_norm
                    )

                    # Save ssim_maps
                    img_SSIM_GTvsPrediction_32bit = np.float32(img_SSIM_GTvsPrediction)
                    io.imsave(
                        os.path.join(
                            QC_model_path,
                            QC_model_name,
                            "SSIM_GTvsPrediction_",
                            i,
                        ),
                        img_SSIM_GTvsPrediction_32bit,
                    )

                    img_SSIM_GTvsSource_32bit = np.float32(img_SSIM_GTvsSource)
                    io.imsave(
                        os.path.join(
                            QC_model_path,
                            QC_model_name,
                            "SSIM_GTvsSource_",
                            i,
                        ),
                        img_SSIM_GTvsSource_32bit,
                    )

                    # Calculate the Root Squared Error (RSE) maps
                    img_RSE_GTvsPrediction = np.sqrt(
                        np.square(test_GT_norm - test_prediction_norm)
                    )
                    img_RSE_GTvsSource = np.sqrt(np.square(test_GT_norm - test_source_norm))

                    # Save SE maps
                    img_RSE_GTvsPrediction_32bit = np.float32(img_RSE_GTvsPrediction)
                    img_RSE_GTvsSource_32bit = np.float32(img_RSE_GTvsSource)
                    io.imsave(
                        os.path.join(
                            QC_model_path,
                            QC_model_name,
                            "RSE_GTvsPrediction_",
                            i,
                        ),
                        img_RSE_GTvsPrediction_32bit,
                    )
                    io.imsave(
                        os.path.join(
                            QC_model_path,
                            QC_model_name,
                            # "Quality Control",
                            "RSE_GTvsSource_",
                            i,
                        ),
                        img_RSE_GTvsSource_32bit,
                    )

                    # -------------------------------- Calculate the RSE metrics and save them --------------------------------

                    # Normalised Root Mean Squared Error (here it's valid to take the mean of the image)
                    NRMSE_GTvsPrediction = np.sqrt(np.mean(img_RSE_GTvsPrediction))
                    NRMSE_GTvsSource = np.sqrt(np.mean(img_RSE_GTvsSource))

                    # We can also measure the peak signal to noise ratio between the images
                    PSNR_GTvsPrediction = psnr(
                        test_GT_norm, test_prediction_norm, data_range=1.0
                    )
                    PSNR_GTvsSource = psnr(test_GT_norm, test_source_norm, data_range=1.0)

                    writer.writerow(
                        [
                            i,
                            str(index_SSIM_GTvsPrediction),
                            str(index_SSIM_GTvsSource),
                            str(NRMSE_GTvsPrediction),
                            str(NRMSE_GTvsSource),
                            str(PSNR_GTvsPrediction),
                            str(PSNR_GTvsSource),
                        ]
                    )

                # error_mapping_report(
                #         Target_QC_folder,
                #         Source_QC_folder,
                #         QC_model_path,
                #         QC_model_name,
                #         img_SSIM_GTvsPrediction,
                #         index_SSIM_GTvsSource,
                #         img_SSIM_GTvsSource,
                #         index_SSIM_GTvsPrediction,
                #         NRMSE_GTvsSource,
                #         PSNR_GTvsSource,
                #         img_RSE_GTvsSource,
                #         NRMSE_GTvsPrediction,
                #         PSNR_GTvsPrediction,
                #         img_RSE_GTvsPrediction,
                #     )

            full_QC_model_path = os.path.join(QC_model_path, QC_model_name)
            # All data is now processed saved
            Test_FileList = os.listdir(
                Source_QC_folder
            )  # this assumes, as it should, that both source and target are named the same
            if len(Test_FileList)==0:
                print("No files in QC_folder")
            else:
                plt.figure(figsize=(15, 15))
                # Currently only displays the last computed set, from memory
                # Target (Ground-truth)
                plt.subplot(3, 3, 1)
                plt.axis("off")
                img_GT = io.imread(os.path.join(Target_QC_folder, Test_FileList[-1]))
                plt.imshow(img_GT)
                plt.title("Target", fontsize=15)

                # Source
                plt.subplot(3, 3, 2)
                plt.axis("off")
                img_Source = io.imread(os.path.join(Source_QC_folder, Test_FileList[-1]))
                plt.imshow(img_Source)
                plt.title("Source", fontsize=15)

                # Prediction
                plt.subplot(3, 3, 3)
                plt.axis("off")
                img_Prediction_path = os.path.join(
                    QC_model_path,
                    QC_model_name,
                    # "Quality Control",
                    "Prediction", Test_FileList[-1]
                )
                img_Prediction = io.imread(
                    img_Prediction_path,
                )
                plt.imshow(img_Prediction)
                plt.title("Prediction", fontsize=15)

                # Setting up colours
                cmap = plt.cm.CMRmap

                # SSIM between GT and Source
                plt.subplot(3, 3, 5)
                # plt.axis('off')
                plt.tick_params(
                    axis="both",  # changes apply to the x-axis and y-axis
                    which="both",  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    left=False,  # ticks along the left edge are off
                    right=False,  # ticks along the right edge are off
                    labelbottom=False,
                    labelleft=False,
                )
                imSSIM_GTvsSource = plt.imshow(img_SSIM_GTvsSource, cmap=cmap, vmin=0, vmax=1)
                plt.colorbar(imSSIM_GTvsSource, fraction=0.046, pad=0.04)
                plt.title("Target vs. Source", fontsize=15)
                plt.xlabel("mSSIM: " + str(round(index_SSIM_GTvsSource, 3)), fontsize=14)
                plt.ylabel("SSIM maps", fontsize=20, rotation=0, labelpad=75)

                # SSIM between GT and Prediction
                plt.subplot(3, 3, 6)
                # plt.axis('off')
                plt.tick_params(
                    axis="both",  # changes apply to the x-axis and y-axis
                    which="both",  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    left=False,  # ticks along the left edge are off
                    right=False,  # ticks along the right edge are off
                    labelbottom=False,
                    labelleft=False,
                )
                imSSIM_GTvsPrediction = plt.imshow(
                    img_SSIM_GTvsPrediction, cmap=cmap, vmin=0, vmax=1
                )
                plt.colorbar(imSSIM_GTvsPrediction, fraction=0.046, pad=0.04)
                plt.title("Target vs. Prediction", fontsize=15)
                plt.xlabel("mSSIM: " + str(round(index_SSIM_GTvsPrediction, 3)), fontsize=14)

                # Root Squared Error between GT and Source
                plt.subplot(3, 3, 8)
                # plt.axis('off')
                plt.tick_params(
                    axis="both",  # changes apply to the x-axis and y-axis
                    which="both",  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    left=False,  # ticks along the left edge are off
                    right=False,  # ticks along the right edge are off
                    labelbottom=False,
                    labelleft=False,
                )
                imRSE_GTvsSource = plt.imshow(img_RSE_GTvsSource, cmap=cmap, vmin=0, vmax=1)
                plt.colorbar(imRSE_GTvsSource, fraction=0.046, pad=0.04)
                plt.title("Target vs. Source", fontsize=15)
                plt.xlabel(
                    "NRMSE: "
                    + str(round(NRMSE_GTvsSource, 3))
                    + ", PSNR: "
                    + str(round(PSNR_GTvsSource, 3)),
                    fontsize=14,
                )
                # plt.title('Target vs. Source PSNR: '+str(round(PSNR_GTvsSource,3)))
                plt.ylabel("RSE maps", fontsize=20, rotation=0, labelpad=75)

                # Root Squared Error between GT and Prediction
                plt.subplot(3, 3, 9)
                # plt.axis('off')
                plt.tick_params(
                    axis="both",  # changes apply to the x-axis and y-axis
                    which="both",  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    left=False,  # ticks along the left edge are off
                    right=False,  # ticks along the right edge are off
                    labelbottom=False,
                    labelleft=False,
                )
                imRSE_GTvsPrediction = plt.imshow(img_RSE_GTvsPrediction, cmap=cmap, vmin=0, vmax=1)
                plt.colorbar(imRSE_GTvsPrediction, fraction=0.046, pad=0.04)
                plt.title("Target vs. Prediction", fontsize=15)
                plt.xlabel(
                    "NRMSE: "
                    + str(round(NRMSE_GTvsPrediction, 3))
                    + ", PSNR: "
                    + str(round(PSNR_GTvsPrediction, 3)),
                    fontsize=14,
                )
                QC_example_data_path = os.path.join(
                    QC_model_path, QC_model_name, "QC_example_data.png"
                )
                plt.savefig(QC_example_data_path, bbox_inches="tight", pad_inches=0)
        except FileNotFoundError:
            print("No prediction example")


def error_mapping_report(
    Target_QC_folder,
    Source_QC_folder,
    QC_model_path,
    QC_model_name,
    img_SSIM_GTvsPrediction,
    index_SSIM_GTvsSource,
    img_SSIM_GTvsSource,
    index_SSIM_GTvsPrediction,
    NRMSE_GTvsSource,
    PSNR_GTvsSource,
    img_RSE_GTvsSource,
    NRMSE_GTvsPrediction,
    PSNR_GTvsPrediction,
    img_RSE_GTvsPrediction,
):
    full_QC_model_path = os.path.join(QC_model_path, QC_model_name)
    # All data is now processed saved
    Test_FileList = os.listdir(
        Source_QC_folder
    )  # this assumes, as it should, that both source and target are named the same

    plt.figure(figsize=(15, 15))
    # Currently only displays the last computed set, from memory
    # Target (Ground-truth)
    plt.subplot(3, 3, 1)
    plt.axis("off")
    img_GT = io.imread(os.path.join(Target_QC_folder, Test_FileList[-1]))
    plt.imshow(img_GT)
    plt.title("Target", fontsize=15)

    # Source
    plt.subplot(3, 3, 2)
    plt.axis("off")
    img_Source = io.imread(os.path.join(Source_QC_folder, Test_FileList[-1]))
    plt.imshow(img_Source)
    plt.title("Source", fontsize=15)

    # Prediction
    plt.subplot(3, 3, 3)
    plt.axis("off")
    img_Prediction_path = os.path.join(
        QC_model_path, QC_model_name, "Prediction", Test_FileList[-1]
    )
    img_Prediction = io.imread(
        img_Prediction_path,
    )
    plt.imshow(img_Prediction)
    plt.title("Prediction", fontsize=15)

    # Setting up colours
    cmap = plt.cm.CMRmap

    # SSIM between GT and Source
    plt.subplot(3, 3, 5)
    # plt.axis('off')
    plt.tick_params(
        axis="both",  # changes apply to the x-axis and y-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,  # ticks along the left edge are off
        right=False,  # ticks along the right edge are off
        labelbottom=False,
        labelleft=False,
    )
    imSSIM_GTvsSource = plt.imshow(img_SSIM_GTvsSource, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(imSSIM_GTvsSource, fraction=0.046, pad=0.04)
    plt.title("Target vs. Source", fontsize=15)
    plt.xlabel("mSSIM: " + str(round(index_SSIM_GTvsSource, 3)), fontsize=14)
    plt.ylabel("SSIM maps", fontsize=20, rotation=0, labelpad=75)

    # SSIM between GT and Prediction
    plt.subplot(3, 3, 6)
    # plt.axis('off')
    plt.tick_params(
        axis="both",  # changes apply to the x-axis and y-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,  # ticks along the left edge are off
        right=False,  # ticks along the right edge are off
        labelbottom=False,
        labelleft=False,
    )
    imSSIM_GTvsPrediction = plt.imshow(
        img_SSIM_GTvsPrediction, cmap=cmap, vmin=0, vmax=1
    )
    plt.colorbar(imSSIM_GTvsPrediction, fraction=0.046, pad=0.04)
    plt.title("Target vs. Prediction", fontsize=15)
    plt.xlabel("mSSIM: " + str(round(index_SSIM_GTvsPrediction, 3)), fontsize=14)

    # Root Squared Error between GT and Source
    plt.subplot(3, 3, 8)
    # plt.axis('off')
    plt.tick_params(
        axis="both",  # changes apply to the x-axis and y-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,  # ticks along the left edge are off
        right=False,  # ticks along the right edge are off
        labelbottom=False,
        labelleft=False,
    )
    imRSE_GTvsSource = plt.imshow(img_RSE_GTvsSource, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(imRSE_GTvsSource, fraction=0.046, pad=0.04)
    plt.title("Target vs. Source", fontsize=15)
    plt.xlabel(
        "NRMSE: "
        + str(round(NRMSE_GTvsSource, 3))
        + ", PSNR: "
        + str(round(PSNR_GTvsSource, 3)),
        fontsize=14,
    )
    # plt.title('Target vs. Source PSNR: '+str(round(PSNR_GTvsSource,3)))
    plt.ylabel("RSE maps", fontsize=20, rotation=0, labelpad=75)

    # Root Squared Error between GT and Prediction
    plt.subplot(3, 3, 9)
    # plt.axis('off')
    plt.tick_params(
        axis="both",  # changes apply to the x-axis and y-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,  # ticks along the left edge are off
        right=False,  # ticks along the right edge are off
        labelbottom=False,
        labelleft=False,
    )
    imRSE_GTvsPrediction = plt.imshow(img_RSE_GTvsPrediction, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(imRSE_GTvsPrediction, fraction=0.046, pad=0.04)
    plt.title("Target vs. Prediction", fontsize=15)
    plt.xlabel(
        "NRMSE: "
        + str(round(NRMSE_GTvsPrediction, 3))
        + ", PSNR: "
        + str(round(PSNR_GTvsPrediction, 3)),
        fontsize=14,
    )
    QC_example_data_path = os.path.join(
        QC_model_path, QC_model_name, "QC_example_data.png"
    )
    plt.savefig(QC_example_data_path, bbox_inches="tight", pad_inches=0)


def quality_tf(history, model_path, model_name, QC_model_name, QC_model_path):
    df = get_history_df_from_model_tf(history)
    df_to_csv(df, model_path, model_name)
    try:
        display_training_errors(model_name, model_path)
    except FileNotFoundError:
        print("Couldn't find loss csv")

    return df


def get_history_df_from_model_tf(history):
    return pd.DataFrame(history)


def full(
    model_path,
    model_name,
    QC_model_name,
    QC_model_path,
    ref_str,
    network,
    Use_the_current_trained_model=True,
    Source_QC_folder=None,
    Target_QC_folder=None,
    show_images=False
):
    full_QC_model_path = os.path.join(QC_model_path, QC_model_name)
    # quality_tf(self, model, model_path, model_name)
    quality_folder_reset(QC_model_path, QC_model_name)
    qc_model_checks(
        QC_model_name,
        QC_model_path,
        model_name,
        model_path,
        Use_the_current_trained_model,
    )
    inspect_loss(QC_model_name, QC_model_path, show_images=show_images)
    if Source_QC_folder is not None:
        create_qc_csv(QC_model_path, QC_model_name, Source_QC_folder, Target_QC_folder)
    reporting.qc_pdf_export(QC_model_name, full_QC_model_path, ref_str, network)
