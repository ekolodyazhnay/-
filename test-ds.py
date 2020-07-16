import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from PIL import Image
from skimage.io import imread
from skimage.transform import resize
import cv2
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,
    ToFloat, ShiftScaleRotate)
from tensorflow.keras.applications.resnet50 import ResNet50

# Начало решения
from fastai import *
from fastai.vision import *
path = Path("FGCC/data/cloth_categories/")
data = ImageDataBunch.from_csv(path, csv_labels="train_labels.csv", ds_tfms=get_transforms(), size=150)
data.normalize(imagenet_stats)
data.show_batch(rows=8, figsize=(14,12))
print(data.classes)
print (len(data.classes),data.c)
learn = create_cnn(data, models.resnet50, metrics=accuracy)
learn.fit_one_cycle(3, 1e-2)
learn.save("stage-1_res-50_sz-150")
learn.show_results(rows=3, figsize=(12,15))
