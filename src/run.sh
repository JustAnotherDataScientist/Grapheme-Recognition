export CUDA_VISIBLE_DEVICES=0,1
export IMAGE_HEIGHT=137
export IMAGE_WIDTH=236
export EPOCHS=3
export TRAIN_BATCH_SIZE=500
export VALID_BATCH_SIZE=8
export MODEL_MEAN="(0.485, 0.456, 0.406)"
export MODEL_STD="(0.229, 0.224, 0.225)"
export BASE_MODEL="ResNet34"
export TRAINING_FOLDS_CSV="../input/train_folds.csv"

export TRAINING_FOLDS="(0, 1, 2, 3)"
export VALID_FOLDS="(4,)"
python train.py

export TRAINING_FOLDS="(0, 1, 2, 4)"
export VALID_FOLDS="(3,)"
python train.py

export TRAINING_FOLDS="(0, 1, 4, 3)"
export VALID_FOLDS="(2,)"
python train.py

export TRAINING_FOLDS="(0, 4, 2, 3)"
export VALID_FOLDS="(1,)"
python train.py

export TRAINING_FOLDS="(4, 1, 2, 3)"
export VALID_FOLDS="(0,)"
python train.py