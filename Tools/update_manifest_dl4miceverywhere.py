from ruamel.yaml import YAML
import sys
import os

def main():
    from parser_dicts_variables import dict_dl4miceverywhere_to_manifest
    dict_manifest_to_dl4miceverywhere = {value: key for key, value in dict_dl4miceverywhere_to_manifest.items()}

    # Read the information from the manifest
    with open('manifest.bioimage.io.yaml', 'r') as f:
        yaml = YAML()
        yaml.preserve_quotes = True
        manifest_data = yaml.load(f)
    
    dl4miceverywhere_notebooks_path = '../DL4MicEverywhere/notebooks/ZeroCostDL4Mic_notebooks'

    # List where the new collection (with versions updates) will be stored
    new_collection = []

    # We are going to check the elements in the collection from the manifest
    for element in manifest_data['collection']:
        new_element = element.copy()
        # In case it is an application, we need to update configuration information with the one on DL4Miceverywhere
        if element['type'] == 'application':
            if element['id'] in dict_manifest_to_dl4miceverywhere.keys(): 
                configuration_path = os.path.join(dl4miceverywhere_notebooks_path, dict_manifest_to_dl4miceverywhere[element['id']], 'configuration.yaml')
                with open(configuration_path, 'r') as f:
                    yaml = YAML()
                    yaml.preserve_quotes = True
                    config_data = yaml.load(f)

                if 'config' not in element:
                    new_element['config'] = config_data['config']
                else:
                    new_element['config']['dl4miceverywhere'] = config_data['config']['dl4miceverywhere']
        
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