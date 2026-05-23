# Face Detection with SVM and HOG Features

Binary face detection system built with classical computer vision techniques - no deep learning required.

## Overview

The model classifies image regions as **face** or **non-face** using HOG (Histogram of Oriented Gradients) features fed into a Support Vector Machine (SVM) with linear kernel. A sliding window approach enables detection on full images at multiple scales.

## Dataset

| Split | Source | Size |
|---|---|---|
| Faces | Labeled Faces in the Wild (LFW) - Kaggle | 5,000 images |
| Non-faces | COCO 2017 Train (filtered: no person category) | 5,000 images |

Non-face images were filtered programmatically using COCO JSON annotations to exclude any image containing a person, ensuring clean negative examples.

## Pipeline

1. **Preprocessing** - grayscale conversion, resize to 150×150, normalization to [0, 1]
2. **Augmentation** - random rotation, scaling, translation and brightness variation (batch processing to control RAM usage)
3. **Feature extraction** - HOG features via scikit-image
4. **Training** - SVM with linear kernel, optimized with GridSearchCV
5. **Inference** - sliding window at multiple scales (75, 50, 25 px) with probability thresholding (default: 0.65)

## Results

Evaluated on 30% hold-out test set (3,000 samples).

| Metric | Score |
|---|---|
| ROC-AUC | **0.9597** |
| Average Precision (AP) | **0.9575** |
| Correct predictions (TP + TN) | **2,683 / 3,000 (89.4%)** |
| False negatives (missed faces) | 174 |
| False positives (non-faces misclassified) | 143 |

## Tech Stack

Python · NumPy · OpenCV · scikit-image · scikit-learn · Matplotlib · joblib

## How to Run

The notebook is optimized for **Google Colab**.

1. Open `face_detection.ipynb` on Google Colab
2. Run cells sequentially
3. The trained model is saved automatically to Google Drive as `face_detection_svm.pkl`

To run inference on a new image, load the saved model:

```python
import joblib
model = joblib.load("face_detection_svm.pkl")
```

## Project Structure

```
face_detection/
└── face_detection.ipynb   # Main notebook (training + inference)
```
