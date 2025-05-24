import os
import shutil
import yaml

def copy_directory(source_dir, destination_dir):
    #if os.path.exists(destination_dir):
    #    raise FileExistsError(f"Destination '{destination_dir}' already exists.")
    shutil.copytree(source_dir, destination_dir)
    print(f"Copied directory from '{source_dir}' to '{destination_dir}'")

def modify_yaml_files(root_dir, key_to_change, new_value):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                yaml_path = os.path.join(subdir, file)
                with open(yaml_path, 'r') as f:
                    try:
                        data = yaml.safe_load(f) or {}
                    except yaml.YAMLError as e:
                        print(f"Error parsing {yaml_path}: {e}")
                        continue

                # Update the key if it exists
                if key_to_change in data:
                    data[key_to_change] = new_value
                    with open(yaml_path, 'w') as f:
                        yaml.safe_dump(data, f)
                    print(f"Updated '{key_to_change}' in {yaml_path}")
                else:
                    print(f"Key '{key_to_change}' not found in {yaml_path}")

if __name__ == "__main__":
    source = './Linear_Representation'
    destination = './Random'
    key = 'classifier_type'         # The YAML key to update
    value = 'Random'           # New value for the key

    copy_directory(source, destination)
    modify_yaml_files(destination, key, value)
