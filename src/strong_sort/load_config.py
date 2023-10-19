
import yaml
import os
def load_config():
    current_working_directory = os.getcwd()
    yaml_path= os.path.join(current_working_directory, 'config.yml')

    with open(yaml_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None