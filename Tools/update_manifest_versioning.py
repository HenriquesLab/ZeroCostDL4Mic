from ruamel.yaml import YAML
import sys
import pandas as pd

from parser_dicts_variables import dict_manifest_to_version

def main():
    # Read the information from the manifest
    with open('manifest.bioimage.io.yaml', 'r', encoding='utf8') as f:
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
            notebook_version = all_notebook_versions[all_notebook_versions["Notebook"] == dict_manifest_to_version[element['id']]]['Version'].iloc[0]
            new_element['version'] = notebook_version
            if 'dl4miceverywhere' not in  element['tags']:
                new_element['tags'].append('dl4miceverywhere')
        
        new_collection.append(new_element)

    # Add the new collection to the manifest
    manifest_data['collection'] = new_collection

    # Write the new collection
    with open('manifest.bioimage.io.yaml', 'w', encoding='utf8') as f:
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
