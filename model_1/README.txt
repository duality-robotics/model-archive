This repository contains a custom YOLOv8 object detection model trained using the Ultralytics framework on the IEEE VisioDECT dataset. The model is designed to detect drones in visual imagery and can be used for applications such as aerial surveillance, airspace monitoring, security systems, and autonomous tracking.


## Dataset

This model was trained on the VisioDECT dataset:

Simeon Okechukwu Ajakwe et al., "VisioDECT Dataset: An Aerial Dataset for
Scenario-Based Multi-Drone Detection and Identification," IEEE DataPort, 2022.
DOI: 10.21227/n27q-7e06
https://ieee-dataport.org/documents/visiodect-dataset-aerial-dataset-scenario-based-multi-drone-detection-and-identification

To reproduce training, download the dataset from the link above and place it
under the following structure:

data/
  images/
    train/
    train_syn/
    val/

