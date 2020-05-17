import joblib
from PIL import Image
import albumentations
import numpy as np
import pretrainedmodels
import torch
import torch.nn.functional as F
import os

# # See dict keys from Image class of PIL module
#
# aug = albumentations.Compose([
#                 albumentations.ShiftScaleRotate(shift_limit=0.0625,
#                                                 scale_limit=0.1,
#                                                 rotate_limit=5,
#                                                 p=0.9)
# ])
# image = joblib.load('./input/image_pickles/Train_68709.pkl')
# image = image.reshape(137, 236).astype(float)
# image = Image.fromarray(image).convert("RGB")
# dict = aug(image=np.array(image))
#
# print(dict.keys())
#
# # See features method from pretrainedmodels module
#
# x = torch.ones((1, 3, 137, 236))
# model = pretrainedmodels.__dict__['resnet34'](pretrained="imagenet")
# x = model.features(x)
# print(x)
# print(x.shape)
# x = F.adaptive_avg_pool2d(x, 1).reshape(1, -1)
# print(x.shape)

# Print strings
print(1)
BASE_MODEL = os.getenv("BASE_MODEL")
print(BASE_MODEL)