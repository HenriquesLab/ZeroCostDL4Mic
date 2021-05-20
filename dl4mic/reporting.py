import numpy as np
from matplotlib import pyplot as plt
import urllib
import os, random
import shutil
import zipfile
from tifffile import imread, imsave
import time
import sys
import wget
from pathlib import Path
import pandas as pd
import csv
from glob import glob
from scipy import signal
from scipy import ndimage
from skimage import io
from sklearn.linear_model import LinearRegression
from skimage.util import img_as_uint
import matplotlib as mpl
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio as psnr
from astropy.visualization import simple_norm
from skimage import img_as_float32
from fpdf import FPDF, HTMLMixin
from datetime import datetime
from pip._internal.operations.freeze import freeze
import subprocess
from datetime import datetime

from . import utils


def pdf_export(
    model_name,
    model_path,
    ref_str,
    ref_aug,
    Notebook_version,
    initial_learning_rate,
    number_of_steps,
    percentage_validation,
    image_patches,
    loss_function,
    batch_size,
    patch_size,
    Training_source,
    number_of_epochs,
    Use_Default_Advanced_Parameters,
    time_start=None,
    example_image=None,
    trained=False,
    augmentation=False,
    Use_pretrained_model=False,
):
    class MyFPDF(FPDF, HTMLMixin):
        pass

    if time_start != None:
        hour, mins, sec = utils.time_elapsed(time_start)
    else:
        hour, mins, sec = [0] * 3

    pdf = MyFPDF()
    pdf.add_page()
    pdf.set_right_margin(-1)
    pdf.set_font("Arial", size=11, style="B")

    Network = "Noise2Void 2D"
    day = datetime.now()
    datetime_str = str(day)[0:10]

    Header = (
        "Training report for "
        + Network
        + " model ("
        + model_name
        + ")\nDate: "
        + datetime_str
    )
    pdf.multi_cell(180, 5, txt=Header, align="L")

    # add another cell
    if trained:
        training_time = (
            "Training time: "
            + str(hour)
            + "hour(s) "
            + str(mins)
            + "min(s) "
            + str(round(sec))
            + "sec(s)"
        )
        pdf.cell(190, 5, txt=training_time, ln=1, align="L")
    pdf.ln(1)

    Header_2 = "Information for your materials and method:"
    pdf.cell(190, 5, txt=Header_2, ln=1, align="L")

    all_packages = ""
    for requirement in freeze(local_only=True):
        all_packages = all_packages + requirement + ", "
    # print(all_packages)

    # Main Packages
    main_packages = ""
    version_numbers = []
    for name in ["tensorflow", "numpy", "Keras", "csbdeep"]:
        find_name = all_packages.find(name)
        main_packages = (
            main_packages
            + all_packages[find_name : all_packages.find(",", find_name)]
            + ", "
        )
        # Version numbers only here:
        version_numbers.append(
            all_packages[find_name + len(name) + 2 : all_packages.find(",", find_name)]
        )

    cuda_version = subprocess.run("nvcc --version", stdout=subprocess.PIPE, shell=True)
    cuda_version = cuda_version.stdout.decode("utf-8")
    cuda_version = cuda_version[cuda_version.find(", V") + 3 : -1]
    gpu_name = subprocess.run("nvidia-smi", stdout=subprocess.PIPE, shell=True)
    gpu_name = gpu_name.stdout.decode("utf-8")
    gpu_name = gpu_name[gpu_name.find("Tesla") : gpu_name.find("Tesla") + 10]
    # if gpu_name == None:
    gpu_name = "CPU"

    # print(cuda_version[cuda_version.find(', V')+3:-1])
    # print(gpu_name)

    shape = io.imread(
        os.path.join(Training_source, os.listdir(Training_source)[0])
    ).shape
    dataset_size = len(os.listdir(Training_source))

    text = (
        "The "
        + str(Network)
        + " model was trained from scratch for "
        + str(number_of_epochs)
        + " epochs on "
        + str(image_patches)
        + " image patches (image dimensions: "
        + str(shape)
        + ", patch size: ("
        + str(patch_size)
        + ","
        + str(patch_size)
        + ")) with a batch size of "
        + str(batch_size)
        + " and a "
        + str(loss_function)
        + " loss function, using the "
        + str(Network)
        + " ZeroCostDL4Mic notebook (v "
        + str(Notebook_version)
        + ") (von Chamier & Laine et al., 2020). Key python packages used include tensorflow (v "
        + str(version_numbers[0])
        + "), Keras (v "
        + str(version_numbers[2])
        + "), csbdeep (v "
        + str(version_numbers[3])
        + "), numpy (v "
        + str(version_numbers[1])
        + "), cuda (v "
        + str(cuda_version)
        + "). The training was accelerated using a "
        + str(gpu_name)
        + "GPU."
    )

    if Use_pretrained_model:
        text = (
            "The "
            + Network
            + " model was trained for "
            + str(number_of_epochs)
            + " epochs on "
            + str(image_patches)
            + " paired image patches (image dimensions: "
            + str(shape)
            + ", patch size: ("
            + str(patch_size)
            + ","
            + str(patch_size)
            + ")) with a batch size of "
            + str(batch_size)
            + " and a "
            + loss_function
            + " loss function, using the "
            + Network
            + " ZeroCostDL4Mic notebook (v "
            + Notebook_version[0]
            + ") (von Chamier & Laine et al., 2020). The model was re-trained from a pretrained model. Key python packages used include tensorflow (v "
            + version_numbers[0]
            + "), Keras (v "
            + version_numbers[2]
            + "), csbdeep (v "
            + version_numbers[3]
            + "), numpy (v "
            + version_numbers[1]
            + "), cuda (v "
            + cuda_version
            + "). The training was accelerated using a "
            + gpu_name
            + "GPU."
        )

    pdf.set_font("")
    pdf.set_font_size(10.0)
    pdf.multi_cell(190, 5, txt=text, align="L")
    pdf.set_font("")
    pdf.set_font("Arial", size=10, style="B")
    pdf.ln(1)
    pdf.cell(26, 5, txt="Augmentation: ", ln=0)
    pdf.set_font("")
    if augmentation:
        aug_text = "The dataset was augmented by default."
    else:
        aug_text = "No augmentation was used for training."
    pdf.multi_cell(190, 5, txt=aug_text, align="L")
    pdf.set_font("Arial", size=11, style="B")
    pdf.ln(1)
    pdf.cell(180, 5, txt="Parameters", align="L", ln=1)
    pdf.set_font("")
    pdf.set_font_size(10.0)
    if Use_Default_Advanced_Parameters:
        pdf.cell(200, 5, txt="Default Advanced Parameters were enabled")
    pdf.cell(200, 5, txt="The following parameters were used for training:")
    pdf.ln(1)
    html = """ 
    <table width=40% style="margin-left:0px;">
      <tr>
        <th width = 50% align="left">Parameter</th>
        <th width = 50% align="left">Value</th>
      </tr>
      <tr>
        <td width = 50%>number_of_epochs</td>
        <td width = 50%>{0}</td>
      </tr>
      <tr>
        <td width = 50%>patch_size</td>
        <td width = 50%>{1}</td>
      </tr>
      <tr>
        <td width = 50%>batch_size</td>
        <td width = 50%>{2}</td>
      </tr>
      <tr>
        <td width = 50%>number_of_steps</td>
        <td width = 50%>{3}</td>
      </tr>
      <tr>
        <td width = 50%>percentage_validation</td>
        <td width = 50%>{4}</td>
      </tr>
      <tr>
        <td width = 50%>initial_learning_rate</td>
        <td width = 50%>{5}</td>
      </tr>
    </table>
    """.format(
        number_of_epochs,
        str(patch_size) + "x" + str(patch_size),
        batch_size,
        number_of_steps,
        percentage_validation,
        initial_learning_rate,
    )
    pdf.write_html(html)

    # pdf.multi_cell(190, 5, txt = text_2, align='L')
    pdf.set_font("Arial", size=11, style="B")
    pdf.ln(1)
    pdf.cell(190, 5, txt="Training Dataset", align="L", ln=1)
    pdf.set_font("")
    pdf.set_font("Arial", size=10, style="B")
    pdf.cell(28, 5, txt="Training_source:", align="L", ln=0)
    pdf.set_font("")
    pdf.multi_cell(170, 5, txt=str(Training_source), align="L")
    # pdf.set_font('')
    # pdf.set_font('Arial', size = 10, style = 'B')
    # pdf.cell(28, 5, txt= 'Training_target:', align = 'L', ln=0)
    # pdf.set_font('')
    # pdf.multi_cell(170, 5, txt = Training_target, align = 'L')
    # pdf.cell(190, 5, txt=aug_text, align='L', ln=1)
    pdf.ln(1)
    pdf.set_font("")
    pdf.set_font("Arial", size=10, style="B")
    pdf.cell(21, 5, txt="Model Path:", align="L", ln=0)
    pdf.set_font("")
    pdf.multi_cell(170, 5, txt=str(model_path) + "/" + str(model_name), align="L")
    pdf.ln(1)
    pdf.cell(60, 5, txt="Example Training Image", ln=1)
    pdf.ln(1)
    if example_image != None:
        exp_size = example_image.shape
        pdf.image(
            example_image,
            x=11,
            y=None,
            w=round(exp_size[1] / 8),
            h=round(exp_size[0] / 8),
        )
        pdf.ln(1)
    ref_1 = 'References:\n - ZeroCostDL4Mic: von Chamier, Lucas & Laine, Romain, et al. "ZeroCostDL4Mic: an open platform to simplify access and use of Deep-Learning in Microscopy." BioRxiv (2020).'
    pdf.multi_cell(190, 5, txt=ref_1, align="L")
    ref_2 = ref_str
    pdf.multi_cell(190, 5, txt=ref_str, align="L")
    if augmentation:
        pdf.multi_cell(190, 5, txt=ref_aug, align="L")
    pdf.ln(3)
    reminder = "Important:\nRemember to perform the quality control step on all newly trained models\nPlease consider depositing your training dataset on Zenodo"
    pdf.set_font("Arial", size=11, style="B")
    pdf.multi_cell(190, 5, txt=reminder, align="C")

    pdf.output(os.path.join(model_path, model_name) + "_training_report.pdf")
    return pdf


def qc_pdf_export(QC_model_name, full_QC_model_path, ref_str, Network):
    class MyFPDF(FPDF, HTMLMixin):
        pass

    pdf = MyFPDF()
    pdf.add_page()
    pdf.set_right_margin(-1)
    pdf.set_font("Arial", size=11, style="B")

    # Network = "Noise2Void 2D"

    day = datetime.now()
    datetime_str = str(day)[0:10]

    Header = (
        "Quality Control report for "
        + Network
        + " model ("
        + QC_model_name
        + ")\nDate: "
        + datetime_str
    )
    pdf.multi_cell(180, 5, txt=Header, align="L")

    all_packages = ""
    for requirement in freeze(local_only=True):
        all_packages = all_packages + requirement + ", "

    pdf.set_font("")
    pdf.set_font("Arial", size=11, style="B")
    pdf.ln(2)
    pdf.cell(190, 5, txt="Development of Training Losses", ln=1, align="L")
    pdf.ln(1)
    if os.path.exists(os.path.join(full_QC_model_path, "lossCurvePlots.png")):
        exp_size = io.imread(
            os.path.join(full_QC_model_path, "lossCurvePlots.png")
        ).shape
        pdf.image(
            os.path.join(full_QC_model_path, "lossCurvePlots.png"),
            x=11,
            y=None,
            w=round(exp_size[1] / 8),
            h=round(exp_size[0] / 8),
        )
    else:
        pdf.set_font("")
        pdf.set_font("Arial", size=10)
        pdf.cell(
            190,
            5,
            txt="If you would like to see the evolution of the loss function during training please play the first cell of the QC section in the notebook.",
        )
    pdf.ln(2)
    pdf.set_font("")
    pdf.set_font("Arial", size=10, style="B")
    pdf.ln(3)
    pdf.cell(80, 5, txt="Example Quality Control Visualisation", ln=1)
    pdf.ln(1)
    try:
        exp_size = io.imread(
            os.path.join(full_QC_model_path, "QC_example_data.png")
        ).shape
        pdf.image(
            os.path.join(full_QC_model_path, "QC_example_data.png"),
            x=16,
            y=None,
            w=round(exp_size[1] / 10),
            h=round(exp_size[0] / 10),
        )
    except FileNotFoundError:
        print("Not QC example image found")

    pdf.ln(1)
    pdf.set_font("")
    pdf.set_font("Arial", size=11, style="B")
    pdf.ln(1)
    pdf.cell(180, 5, txt="Quality Control Metrics", align="L", ln=1)
    pdf.set_font("")
    pdf.set_font_size(10.0)

    pdf.ln(1)
    html = """
  <body>
  <font size="7" face="Courier New" >
  <table width=94% style="margin-left:0px;">"""
    try:
        with open(
            os.path.join(full_QC_model_path, "QC_metrics_" + QC_model_name + ".csv"),
            "r",
        ) as csvfile:
            metrics = csv.reader(csvfile)
            header = next(metrics)
            image = header[0]
            mSSIM_PvsGT = header[1]
            mSSIM_SvsGT = header[2]
            NRMSE_PvsGT = header[3]
            NRMSE_SvsGT = header[4]
            PSNR_PvsGT = header[5]
            PSNR_SvsGT = header[6]
            header = """
        <tr>
        <th width = 10% align="left">{0}</th>
        <th width = 15% align="left">{1}</th>
        <th width = 15% align="center">{2}</th>
        <th width = 15% align="left">{3}</th>
        <th width = 15% align="center">{4}</th>
        <th width = 15% align="left">{5}</th>
        <th width = 15% align="center">{6}</th>
        </tr>""".format(
                image,
                mSSIM_PvsGT,
                mSSIM_SvsGT,
                NRMSE_PvsGT,
                NRMSE_SvsGT,
                PSNR_PvsGT,
                PSNR_SvsGT,
            )
            html = html + header
            for row in metrics:
                image = row[0]
                mSSIM_PvsGT = row[1]
                mSSIM_SvsGT = row[2]
                NRMSE_PvsGT = row[3]
                NRMSE_SvsGT = row[4]
                PSNR_PvsGT = row[5]
                PSNR_SvsGT = row[6]
                cells = """
            <tr>
            <td width = 10% align="left">{0}</td>
            <td width = 15% align="center">{1}</td>
            <td width = 15% align="center">{2}</td>
            <td width = 15% align="center">{3}</td>
            <td width = 15% align="center">{4}</td>
            <td width = 15% align="center">{5}</td>
            <td width = 15% align="center">{6}</td>
            </tr>""".format(
                    image,
                    str(round(float(mSSIM_PvsGT), 3)),
                    str(round(float(mSSIM_SvsGT), 3)),
                    str(round(float(NRMSE_PvsGT), 3)),
                    str(round(float(NRMSE_SvsGT), 3)),
                    str(round(float(PSNR_PvsGT), 3)),
                    str(round(float(PSNR_SvsGT), 3)),
                )
                html = html + cells
            html = html + """</body></table>"""
    except FileNotFoundError:
        print("No qc csv found")
    pdf.write_html(html)

    pdf.ln(1)
    pdf.set_font("")
    pdf.set_font_size(10.0)
    ref_1 = 'References:\n - ZeroCostDL4Mic: von Chamier, Lucas & Laine, Romain, et al. "ZeroCostDL4Mic: an open platform to simplify access and use of Deep-Learning in Microscopy." BioRxiv (2020).'
    pdf.multi_cell(190, 5, txt=ref_1, align="L")
    # ref_2 = '- Noise2Void: Krull, Alexander, Tim-Oliver Buchholz, and Florian Jug. "Noise2void-learning denoising from single noisy images." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.'
    pdf.multi_cell(190, 5, txt=ref_str, align="L")
    pdf.ln(3)
    reminder = "To find the parameters and other information about how this model was trained, go to the training_report.pdf of this model which should be in the folder of the same name."

    pdf.set_font("Arial", size=11, style="B")
    pdf.multi_cell(190, 5, txt=reminder, align="C")

    pdf.output(os.path.join(full_QC_model_path, QC_model_name + "_QC_report.pdf"))
