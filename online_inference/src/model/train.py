#!/usr/bin/python
# -*- coding: UTF-8 -*-
""""
версия 6.0.1

train.py -утилита для тренировки модели на основе resnet18.
В качестве входных данных подается набор маршрутов, проходящих по территории одного класса.
Маршрут задается набором координат точек. Формат - csv
пример:
class x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 x6 y6 x7 y7 x8 y8 x9 y9 x10 y10
0 16227 13054 16300 13017 16286 12929 16117 12904 16105 13024
0 15830 12909
1 19043 11043 18991 11038 19064 10836 19090 10836 19150 10962 19140 10990

"""

import os.path
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from PIL import Image
from omegaconf import DictConfig
import hydra

from utils.transformers import TrainTransformer
from utils.train_utils import train, validate, make_model
from utils.datasets import WalkLinesToDataset

log = logging.getLogger(__name__)


@hydra.main(config_path="conf",
            config_name='train_config.yaml')
def main(cfg: DictConfig) -> None:
    log.info("Start training")
    root_path = os.path.normpath(os.path.join(hydra.utils.get_original_cwd(), cfg.path_to_root))
    Image.MAX_IMAGE_PIXELS = cfg.map.max_image_pixels
    train_transforms = TrainTransformer(cfg.deviation)

    walk_lines = WalkLinesToDataset(walk_file_name=os.path.join(root_path, cfg.walks_file),
                                    image_file_name=os.path.join(root_path, cfg.map.map_image),
                                    transforms=train_transforms,
                                    crop_size=cfg.crop_size,
                                    walk_step=cfg.walk_step)
    dataset_len = len(walk_lines)
    log.info(f"Dataset len {dataset_len}")
    train_len = int(dataset_len * cfg.train_val_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(walk_lines, [train_len, dataset_len - train_len])
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    log.info(f"Working on {device}")

    model = make_model(len(cfg.map.class_list))
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, )
    loss_fn = nn.NLLLoss().to(device)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr,
                                                    steps_per_epoch=int(len(train_data_loader)),
                                                    epochs=cfg.epoch,
                                                    anneal_strategy='linear')

    # 2. train & validate
    log.info("Ready for training...")
    best_val_loss = np.inf

    for epoch in range(cfg.epoch):
        train_loss = train(model, train_data_loader, loss_fn, optimizer, scheduler, device=device)
        val_loss = validate(model, val_data_loader, loss_fn, device=device)
        log.info("Epoch #{:2}:\ttrain loss: {:5.2}\tval loss: {:5.2}".format(epoch, train_loss, val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, os.path.join(root_path, cfg.model_save))

    log.info(f"best val is  {best_val_loss}")
    return 0


if __name__ == "__main__":
    main()
