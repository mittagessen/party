#!/usr/bin/env python
from tqdm import tqdm
from party.dataset import TextLineDataModule

with open('train.lst') as manifest:
    for entry in manifest.readlines():
        im_p = entry.rstrip('\r\n')
        if os.path.isfile(im_p):
            training_files.append(im_p)

with open('val.lst') as manifest:
    for entry in manifest.readlines():
        im_p = entry.rstrip('\r\n')
        if os.path.isfile(im_p):
            val_files.append(im_p)

data_module = TextLineDataModule(training_data=training_files,
                                 evaluation_data=val_files,
                                 prompt_mode='both',
                                 augmentation=True,
                                 batch_size=32,
                                 num_workers=8)

for sample in tqdm(data_module.train_dataloader()):
    pass
