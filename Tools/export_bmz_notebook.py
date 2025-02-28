from bioimageio.core import export_resource_package as save_bioimageio_package
from bioimageio.core import load_resource_description as load_description
# import bioimageio.spec
from pathlib import Path

from ruamel.yaml import YAML
import tempfile
import zipfile
import re
import os

def export_bmz_notebook(notebook_id, output_path):

    zerocostdl4mic_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))

    # Read the information from the manifest
    with open(os.path.join(zerocostdl4mic_path, "manifest.bioimage.io.yaml"), 'r', encoding='utf8') as f:
        yaml = YAML()
        yaml.preserve_quotes = True
        manifest_data = yaml.load(f)

    wanted_notebook = None
    # We are going to check the elements in the collection from the manifest
    for element in manifest_data['collection']:
        # We are only looking for the selected notebook
        if element['id'] == notebook_id: 
            wanted_notebook = element.copy()
            break
    
    # Check if the notebook was found, otherwise raise an Error
    if wanted_notebook is None:
        raise ValueError(f"Sorry, we were not able to find a notebook with the name: {notebook_id}")

    # Get the path to the notebook
    notebook_url = wanted_notebook['config']['dl4miceverywhere']['notebook_url']
    notebook_name = os.path.basename(notebook_url)
    notebook_local_path = os.path.join(zerocostdl4mic_path, "Colab_notebooks", notebook_name)

    # Get the path to the requirements
    requirements_url = wanted_notebook['config']['dl4miceverywhere']['requirements_url']
    requirements_name = os.path.basename(requirements_url)
    requirements_local_path = os.path.join(zerocostdl4mic_path, "requirements_files", requirements_name)

    # Add these files into the attachments:files:[] on notebook the configuration
    if 'attachments' not in wanted_notebook:
        wanted_notebook['attachments'] = {'files': [notebook_local_path, requirements_local_path]} #'requirements.txt']}
    else:
        wanted_notebook['attachments']['files'] = [notebook_local_path, requirements_local_path] #'requirements.txt']

    # Add the bioimageio.spec format version
    # format_version = re.match(r'^(\d+\.\d+\.\d+)', bioimageio.spec.__version__).group(1)
    wanted_notebook['format_version'] = "0.2.4" # format_version # Only X.X.X

    # Add the "zero/" collection at the beginning of the id
    wanted_notebook['id'] = f"zero/{wanted_notebook['id']}"

    # Create rdf.yaml file on a temporary folder
    with tempfile.TemporaryDirectory() as tmpdirname:
        print(f"Temporary directory created on: {tmpdirname}")

        rdf_path = os.path.join(tmpdirname, "rdf.yaml")

        # Write the new collection
        with open(rdf_path, 'w', encoding='utf8') as f:
            yaml = YAML()
            # Choose the settings
            yaml.preserve_quotes = True
            yaml.default_flow_style = False
            yaml.indent(mapping=2, sequence=4, offset=2)
            yaml.width = 10e10
            # Save notebook's data
            yaml.dump(wanted_notebook, f)

        print("The 'rdf.yaml' file was correctly created.")

        os.makedirs(output_path, exist_ok=True)
        zipfile_path = os.path.join(output_path, f"{notebook_id}.zip")
        
        application_description = load_description(Path(rdf_path))
        exported = save_bioimageio_package(
            application_description, output_path=Path(zipfile_path)
        )

        print(f"ZIP file correctly created on: {exported.absolute()}")
        from zipfile import ZipFile 
  
        # loading the temp.zip and creating a zip object 
        with ZipFile(exported.absolute(), 'r') as zObject: 
            # Extracting all the members of the zip into a specific location. 
            zObject.extractall(path=os.path.join(output_path, f"{notebook_id}_unzipped")) 
    
        print(f"Unzipped folder correctly created on: {os.path.join(output_path, f'{notebook_id}_unzipped')}")

def main():
    import argparse
 
    parser = argparse.ArgumentParser(description="Export the chosen notebook into a ZIP file that follows the BioImage.IO format.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--name", help="Name of the notebook you want to export. You need to provide the 'id' from manifest.bioimage.io.yaml.")
    parser.add_argument("-o", "--output", help="Path to the folder to save the ZIP file.")
    args = vars(parser.parse_args())

    export_bmz_notebook(notebook_id=args['name'], output_path=args['output'])

if __name__ == "__main__":
    main()