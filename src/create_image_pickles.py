import pandas as pd
import numpy
import joblib
import glob
from tqdm import tqdm

# Script for dumping pickles of train images.
if __name__ == '__main__':
    files = glob.glob('../input/train_*.parquet')
    for f in files:
        df = pd.read_parquet(f)
        image_ids = df['image_id'].values
        df = df.drop('image_id', axis=1)
        image_arr = df.values
        for j, img_id in tqdm(enumerate(image_ids), total=len(image_ids)):
            filename = "../input/image_pickles/"+str(img_id)+".pkl"
            joblib.dump(image_arr[j, :], filename)
