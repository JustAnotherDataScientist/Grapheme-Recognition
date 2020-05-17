import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# Script for creating stratified kfold csv from train set. Creates '../input/train_folds.csv'.
if __name__ == '__main__':
    df = pd.read_csv("../input/train.csv")
    df.loc[:, 'kfold']=-1
    df = df.sample(frac=1).reset_index(drop=True)

    mlkf = MultilabelStratifiedKFold(n_splits=5)

    X = df['image_id'].values
    y = df[['grapheme_root','vowel_diacritic','consonant_diacritic']].values

    for fold, (train_f_ind, valid_f_ind) in enumerate(mlkf.split(X,y)):
        df.loc[valid_f_ind, 'kfold'] = fold

    df.to_csv("../input/train_folds.csv", index=False)

