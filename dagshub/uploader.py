import shutil
import os
from dagshub import upload_files
from dagshub.data_engine import datasources # type: ignore

def upload_to_dagshub_from_dirs(dir_paths, repo_name, datasource_name='my-datasource', target_folder='data'):
    """ 
    Takes a list of directory paths, collects them into one folder, and uploads it to DagsHub.

    Parameters:
    - dir_paths: List of str - Paths to directories you want to include
    - repo_name: str - Format should be 'username/repo_name'
    - datasource_name: str - DagsHub datasource name (default: 'my-datasource')
    - target_folder: str - Local folder to gather data into before upload (default: 'data')
    """
    os.makedirs(target_folder, exist_ok=True)
    
    # Copy all contents of each directory into the target folder
    for dir_path in dir_paths:
        if not os.path.isdir(dir_path):
            raise ValueError(f"'{dir_path}' is not a valid directory")
        for item in os.listdir(dir_path):
            s = os.path.join(dir_path, item)
            d = os.path.join(target_folder, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

    # Create DagsHub datasource and upload
    datasources.create_datasource(repo_name, datasource_name, target_folder)
    upload_files(repo_name, target_folder)
    print(f"Upload to {repo_name} complete from {target_folder}")
    
    return datasource_name
