from ruamel.yaml import YAML
import sys
import os
import pandas as pd

dict_ID_manifest_verseion = {
                    'Notebook_Augmentor_ZeroCostDL4Mic': 'Augmentor' ,
                    'Notebook_CARE_2D_ZeroCostDL4Mic': 'CARE (2D)' ,
                    'Notebook_CARE_3D_ZeroCostDL4Mic': 'CARE (3D)' ,
                    'Notebook_Cellpose_2D_ZeroCostDL4Mic': 'Cellpose' ,
                    'Notebook_CycleGAN_2D_ZeroCostDL4Mic': 'CycleGAN' ,
                    'Notebook_DFCAN_ZeroCostDL4Mic': 'DFCAN 2D' ,
                    'Notebook_DRMIME_ZeroCostDL4Mic': 'DRMIME' ,
                    'Notebook_DecoNoising_2D_ZeroCostDL4Mic': 'DecoNoising' ,
                    'Notebook_Deep-STORM_2D_ZeroCostDL4Mic': 'Deep-STORM' ,
                    'Notebook_Deep-STORM_2D_ZeroCostDL4Mic_DeepImageJ': 'Deep-STORM BioimageIO' ,
                    'Notebook_DenoiSeg_2D_ZeroCostDL4Mic': 'DenoiSeg' ,
                    'Notebook_Detectron2_ZeroCostDL4Mic': 'Detectron 2D' ,
                    'Notebook_EmbedSeg_2D_ZeroCostDL4Mic': 'EmbedSeg 2D' ,
                    'Notebook_Interactive_Segmentation_Kaibu_2D_ZeroCostDL4Mic': 'Kaibu' ,
                    'Notebook_MaskRCNN_ZeroCostDL4Mic': 'MaskRCNN' ,
                    'Notebook_Noise2Void_2D_ZeroCostDL4Mic': 'Noise2Void (2D)' ,
                    'Notebook_Noise2Void_3D_ZeroCostDL4Mic': 'Noise2Void (3D)' ,
                    'Notebook_Quality_Control_ZeroCostDL4Mic': 'Quality_control' ,
                    'Notebook_RCAN_3D_ZeroCostDL4Mic': '3D RCAN' ,
                    'Notebook_RetinaNet_ZeroCostDL4Mic': 'RetinaNet' ,
                    'Notebook_SplineDist_2D_ZeroCostDL4Mic': 'SplineDist (2D)' ,
                    'Notebook_StarDist_2D_ZeroCostDL4Mic': 'StarDist 2D' ,
                    'Notebook_StarDist_3D_ZeroCostDL4Mic': 'StarDist 3D' ,
                    'Notebook_U-Net_2D_ZeroCostDL4Mic': 'U-Net (2D)', 
                    'Notebook_U-Net_2D_multilabel_ZeroCostDL4Mic': 'U-Net (2D) multilabel',
                    'Notebook_U-Net_3D_ZeroCostDL4Mic': 'U-Net (3D)' ,
                    'Notebook_YOLOv2_ZeroCostDL4Mic': 'YOLOv2' ,
                    'Notebook_fnet_2D_ZeroCostDL4Mic': 'fnet (2D)' ,
                    'Notebook_fnet_3D_ZeroCostDL4Mic': 'fnet (3D)' ,
                    'Notebook_pix2pix_2D_ZeroCostDL4Mic': 'pix2pix' ,
                    'WGAN_ZeroCostDL4Mic.ipynb': 'WGAN 2D' 
                    }

def main():
    # Read the information from the manifest
    with open('manifest.bioimage.io.yaml', 'r') as f:
        yaml = YAML()
        yaml.preserve_quotes = True
        manifest_data = yaml.load(f)
        
    # Read the versions of the notebooks
    all_notebook_versions = pd.read_csv('Colab_notebooks/Latest_Notebook_versions.csv', dtype=str)

    # List where the new collection (with versions updates) will be stored
    new_collection = []

    # We are going to check the elements in the collection from the manifest
    for element in manifest_data['collection']:
        new_element = element.copy()
        # In case it is an application, we need to update the version
        if element['type'] == 'application' and element['id'] != 'notebook_preview': 
            # Check the version based on the ID of the notebook
            notebook_version = all_notebook_versions[all_notebook_versions["Notebook"] == dict_ID_manifest_verseion[element['id']]]['Version'].iloc[0]
            new_element['version'] = notebook_version
        
        new_collection.append(new_element)

    # Add the new collection to the manifest
    manifest_data['collection'] = new_collection

    # Write the new collection
    with open('manifest.bioimage.io.yaml', 'w') as f:
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.default_flow_style = False
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.width = 10e10
        yaml.dump(manifest_data, f)

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        sys.exit(main())
    else:
        sys.exit(1)