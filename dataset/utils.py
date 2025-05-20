import shutil
import os

def merge_folders(src_dir, dest_dir):
    for split in ['train', 'valid', 'test']:
        for data_type in ['images', 'labels']:
            src_path = os.path.join(src_dir, split, data_type)
            dest_path = os.path.join(dest_dir, split, data_type)
            os.makedirs(dest_path, exist_ok=True)
            for file in os.listdir(src_path):
                shutil.copy(os.path.join(src_path, file), os.path.join(dest_path, file))

