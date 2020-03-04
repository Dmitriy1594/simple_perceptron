import csv
import os
import shutil
import random
from math import tanh, cos

def data_to_scv(path_with_name, data):
    with open(f'{path_with_name}', 'w', newline='') as csvfile:
        fieldnames = ['k', 'w', 'y', 'E']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for d in data:
            writer.writerow(d)

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)

        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
