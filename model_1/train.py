import yaml
import os
import cv2
import random
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import ultralytics
from ultralytics import YOLO
import csv
import shutil
import torch 

def generate_yaml(path, train_dirs, val_dirs, nc=1, output_file="data.yaml"):
    if not isinstance(train_dirs, list):
        train_dirs = [train_dirs]
    if not isinstance(val_dirs, list):
        val_dirs = [val_dirs]
    # Construct the data dictionary
    data = {
        "nc": nc,
        "path": path,  # dataset root dir
        "train": [f"{train_dir}" for train_dir in train_dirs],
        "val": [f"{val_dir}" for val_dir in val_dirs],
    }
    
    # Write the data to a YAML file
    with open(output_file, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
        
this_dir = os.getcwd()
path = os.path.join(this_dir, 'data')

train = 'images/train'
val = 'images/val'
name_baseline = 'real'
nc = 1
n_epochs = 10

generate_yaml(path, train, val, nc=nc)
    
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Set training parameters
training_params = {
    'imgsz': 640,          # Image size
    'epochs': n_epochs,             # Number of epochs
    'name': name_baseline,   # Experiment name,
    'optimizer': 'AdamW',
    'lr0': 0.0001,
    'lrf': 0.0001,
    'momentum': 0.9,
    'exist_ok' : True,
    'single_cls' : True,
    'verbose' : False, 
    'cache': False,
}
    
# Train the model
results = model.train(data='data.yaml', **training_params)
results_real = results.box.map50

train = 'images/train_syn'
val = 'images/val'
name_baseline = 'synthetic'
n_epochs = 10

generate_yaml(path, train, val, nc=nc)
    
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# Set training parameters
training_params = {
    'imgsz': 640,          # Image size
    'epochs': n_epochs,             # Number of epochs
    'name': name_baseline,   # Experiment name,
    'optimizer': 'AdamW',
    'lr0': 0.0001,
    'lrf': 0.0001,
    'momentum': 0.9,
    'exist_ok' : True,
    'single_cls' : True,
    'verbose' : False, 
    'cache': False,
}
    
# Train the model
results = model.train(data='data.yaml', **training_params)
results_syn = results.box.map50

train = ['images/train_syn','images/train']
val = 'images/val'
name_baseline = 'mix'
n_epochs = 10

# data_yaml = dict(
#     path = path,
#     train = train,
#     val = val,
#     nc = 1,
#     names =['drone'])

# with open('data.yaml', 'w') as outfile:
#     yaml.dump(data_yaml, outfile, default_flow_style=True)

generate_yaml(path, train, val, nc=nc)
    
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# Set training parameters
training_params = {
    'imgsz': 640,          # Image size
    'epochs': n_epochs,             # Number of epochs
    'name': name_baseline,   # Experiment name,
    'optimizer': 'AdamW',
    'lr0': 0.0001,
    'lrf': 0.0001,
    'momentum': 0.9,
    'exist_ok' : True,
    'single_cls' : True,
    'verbose' : False, 
    'cache': False,
}
    
# Train the model
results = model.train(data='data.yaml', **training_params)
results_mix = results.box.map50

print(f'mAP@50 (real) = {results_real}')
print(f'mAP@50 (syn) = {results_syn}')
print(f'mAP@50 (mix) = {results_mix}')