
import os
import json

def create_dir(path):
    # checking if the directory demo_folder2
    # exist or not.
    if not os.path.isdir(path):
        # if the demo_folder2 directory is
        # not present then create it.
        os.makedirs(path)
    return True

def __load_json__(path):
    with open(path, 'r') as f:
        tmp = json.loads(f.read())

    return tmp