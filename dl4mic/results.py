import pandas as pd
import os
import shutil
import time
from . import pdf_export
import csv

def tf_history_convert(history):
    lossData = pd.DataFrame(history.history)
def torch_history_convert(history):
    pass

def df_history_to_report(lossData,model_path,model_name,history,start,model):
    if os.path.exists(model_path+"/"+model_name+"/Quality Control"):
        shutil.rmtree(model_path+"/"+model_name+"/Quality Control")

    os.makedirs(model_path+"/"+model_name+"/Quality Control")

    # The training evaluation.csv is saved (overwrites the Files if needed). 
    lossDataCSVpath = model_path+'/'+model_name+'/Quality Control/training_evaluation.csv'
    with open(lossDataCSVpath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['loss','val_loss', 'learning rate'])
        for i in range(len(history.history['loss'])):
            writer.writerow([history.history['loss'][i], history.history['val_loss'][i], history.history['lr'][i]])


    # Displaying the time elapsed for training
    dt = time.time() - start
    mins, sec = divmod(dt, 60) 
    hour, mins = divmod(mins, 60) 
    print("Time elapsed:",hour, "hour(s)",mins,"min(s)",round(sec),"sec(s)")

def tf_model_export(model,model_name,model_description,patch_size,X_val,Use_pretrained_model,authors=["You"]):
    model.export_TF(name=model_name,
                    description=model_description, 
                    authors=authors,
                    test_img=X_val[0,...,0], axes='YX',
                    patch_shape=(patch_size, patch_size))

    print("Your model has been sucessfully exported and can now also be used in the CSBdeep Fiji plugin")

    pdf_export(trained = True, pretrained_model = Use_pretrained_model)

def torch_model_export():
    pass

