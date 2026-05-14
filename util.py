import os 
from os import path 


def clear_directory(directory: os.PathLike):
    """Clears the directory of all files."""
    if path.exists(directory):
        for file in os.listdir(directory):
            file_path = path.join(directory, file)
            if path.isfile(file_path):
                os.remove(file_path)