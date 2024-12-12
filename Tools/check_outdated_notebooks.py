from ruamel.yaml import YAML
import pandas as pd

from parser_dicts_variables import dict_manifest_to_version

def check_outdated_notebooks():
    # Read the information from the manifest
    with open('manifest.bioimage.io.yaml', 'r', encoding='utf8') as f:
        yaml = YAML()
        yaml.preserve_quotes = True
        manifest_data = yaml.load(f)
        
    # Read the versions of the notebooks
    all_notebook_versions = pd.read_csv('Colab_notebooks/Latest_Notebook_versions.csv', dtype=str)

    list_notebook_need_update = []
    # We are going to check the elements in the collection from the manifest
    for element in manifest_data['collection']:
        # In case it is an application, we need to update the version
        if element['type'] == 'application' and element['id'] != 'notebook_preview': 
            # Get the different versions of the notebooks (latest_version and manifest)
            notebook_version = all_notebook_versions[all_notebook_versions["Notebook"] == dict_manifest_to_version[element['id']]]['Version'].iloc[0]
            manifest_version = element['version']
            
            # Check if it is the same or different
            if notebook_version != manifest_version:
                # In case its different, add its id to the list
                list_notebook_need_update.append(element['id'])

    return list_notebook_need_update

def main():
    import argparse
 
    parser = argparse.ArgumentParser(description="Checks the notebooks that need to be updated and returns a list with its IDs, no arguments required.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = vars(parser.parse_args())

    list_notebook_need_update = check_outdated_notebooks()
    
    if len(list_notebook_need_update) == 0:
        print('')
    else:
        print(' '.join(list_notebook_need_update))

if __name__ == "__main__":
    main()