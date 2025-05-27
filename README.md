# Face Detection System for a Digital Camera
## Project Context
ProCam S.p.A. is ready to launch a new compact digital camera, accessible and designed for young photography enthusiasts. The main goal of the product is to facilitate the shooting experience, especially for selfies with one or more people.

## Challenge
You have been hired as a Data Scientist to develop a system for detecting faces in images, which will help engineers automatically optimize camera settings during selfies. Your task is to create a pipeline that identifies faces in images and returns the coordinates of the bounding boxes where faces are located. If there are no faces, the pipeline will return an empty list.

This is a Computer Vision problem, more precisely Face Detection.

## Project Requirements
1. Objective: Build a face detection system using Scikit-learn. The pipeline must be able to:

 - Take an input image.
- Return a list of bounding box coordinates where faces are present.
- Return an empty list if there are no faces in the image.

2. Limitations:

- Dataset: You are not provided with a dataset. You must search for a suitable dataset online or, in the absence of alternatives, build it yourself.
- Pre-trained models: You are not allowed to use pre-trained models. The Face Detection model must be trained from scratch with Scikit-learn.
- Compute resources: You will be working on a system with limited computational capabilities. The model must be optimized to use few resources.
- Development environment: Google Colab
