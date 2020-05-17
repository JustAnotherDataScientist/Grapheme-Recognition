from modeldispatcher import MODEL_DISPATCHER
import os
from dataset import BengaliDatasetTrain
import ast
import torch
import torch.nn as nn
from tqdm import tqdm

DEVICE = "cuda"
EPOCHS = int(os.getenv("EPOCHS"))
BASE_MODEL = os.getenv("BASE_MODEL")

TRAINING_FOLDS = ast.literal_eval(os.getenv("TRAINING_FOLDS"))
VALID_FOLDS = ast.literal_eval(os.getenv("VALID_FOLDS"))

IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT"))
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH"))

MODEL_MEAN = ast.literal_eval((os.getenv("MODEL_MEAN")))
MODEL_STD = ast.literal_eval((os.getenv("MODEL_STD")))

TRAIN_BATCH_SIZE = int(os.getenv('TRAIN_BATCH_SIZE'))
VALID_BATCH_SIZE = int(os.getenv('VALID_BATCH_SIZE'))


def loss_fn(outputs, targets):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)
    return (l1+l2+l3)/3


def train_fn(dataset, dataloader, model, optimizer):
    model.train()

    for bi, d in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        image = d['image']
        grapheme_root = d['grapheme_root']
        vowel_diacritic = d['vowel_diacritic']
        consonant_diacritic = d['consonant_diacritic']

        image = image.to(DEVICE, torch.float)
        grapheme_root = grapheme_root.to(DEVICE, torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE, torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE, torch.long)

        optimizer.zero_grad()
        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()


def eval_fn(dataset, dataloader, model):
    model.eval()
    final_loss = 0
    counter = 0
    with torch.no_grad():
        for bi, d in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            image = d['image']
            grapheme_root = d['grapheme_root']
            vowel_diacritic = d['vowel_diacritic']
            consonant_diacritic = d['consonant_diacritic']

            image = image.to(DEVICE, torch.float)
            grapheme_root = grapheme_root.to(DEVICE, torch.long)
            vowel_diacritic = vowel_diacritic.to(DEVICE, torch.long)
            consonant_diacritic = consonant_diacritic.to(DEVICE, torch.long)

            outputs = model(image)
            targets = (grapheme_root, vowel_diacritic, consonant_diacritic)

            loss = loss_fn(outputs, targets)
            final_loss += loss
    return final_loss/counter


def main():
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)
    model.to(DEVICE)

    train_dataset = BengaliDatasetTrain(
        folds=TRAINING_FOLDS,
        img_height=IMAGE_HEIGHT,
        img_width=IMAGE_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    valid_dataset = BengaliDatasetTrain(
        folds=VALID_FOLDS,
        img_height=IMAGE_HEIGHT,
        img_width=IMAGE_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )
    valid_dataloader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           patience=5, factor=0.3)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Can put EarlyStopping there

    for epoch in range(EPOCHS):
        train_fn(train_dataset, train_dataloader, model, optimizer)
        val_score = eval_fn(valid_dataset, valid_dataloader, model)
        scheduler.step(val_score)
        torch.save(model.state_dict(), "{}_{}.bin".format(BASE_MODEL, VALID_FOLDS[0]))


if __name__ == '__main__':
    main()