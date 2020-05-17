import joblib
from PIL import Image
import albumentations
import numpy as np

aug = albumentations.Compose([
                albumentations.ShiftScaleRotate(shift_limit=0.0625,
                                                scale_limit=0.1,
                                                rotate_limit=5,
                                                p=0.9)
])
image = joblib.load('./input/image_pickles/Train_68709.pkl')
image = image.reshape(137, 236).astype(float)
image = Image.fromarray(image).convert("RGB")
dict = aug(image=np.array(image))

print(dict.keys())