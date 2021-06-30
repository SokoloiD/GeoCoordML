#!/usr/bin/python
# -*- coding: UTF-8 -*-
""""
версия 1.0.1

"""

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
import tqdm


def train(model, loader, loss_fn, optimizer, scheduler, device):
    model.train()
    train_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="training..."):
        images = batch["image"].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
        targets = batch["targets"]  # B x (2 * NUM_PTS)

        pred_targets = model(images).cpu()  # B x 2
        loss = loss_fn(pred_targets, targets)
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return np.mean(train_loss)


def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="validation...", position=0, leave=True):
        images = batch["image"].to(device)
        targets = batch["targets"]

        with torch.no_grad():
            pred_targets = model(images).cpu()
        loss = loss_fn(pred_targets, targets)
        val_loss.append(loss.item())

    return np.mean(val_loss)


def make_model(num_labels: int):
    model = resnet18()
    classifier = nn.Sequential(nn.Linear(model.fc.in_features, 512),
                               nn.ReLU(),
                               nn.Linear(512, num_labels),
                               nn.LogSoftmax(dim=1))
    model.fc = classifier
    return model
