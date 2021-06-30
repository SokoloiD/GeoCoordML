#!/usr/bin/python
# -*- coding: UTF-8 -*-
""""
версия 1.0.1
общие утилиты
"""
import os
import logging
import numpy as np
import torch
import functools
import operator
import tqdm

from utils.data_structures import Config, PredictResultDescription

log = logging.getLogger(__name__)


def batch_predict_area_class(model, data_loader, cfg: Config):
    """

    :param model:
    :param data_loader:
    :param cfg:
    :return: кортеж из трех  массивов -
        (координаты исходыне (GPS) в текстовом формате строками размер - N
         координаты точек в пикселях (np.array shape (N, 2)
         предсказания классов np.array shape (N, кол-во классов)

    """
    gps = []
    pixels = []
    predicts = []
    log.info(f"Ready to predict for {len(data_loader)} batches")
    for batch in tqdm.tqdm(data_loader,
                           total=len(data_loader), desc="recognizing...",
                           position=0, unit="points", unit_scale=cfg.batch_size,
                           leave=True):
        with torch.no_grad():
            predicts.append(model(batch["tensor"]).cpu().numpy())
            gps.append(batch["gps"])
            pixels.append(batch["pixels"].numpy())
    gps = functools.reduce(operator.iconcat, gps, [])
    pixels = np.concatenate(pixels, axis=0)
    predicts = np.concatenate(predicts, axis=0)
    log.info(f"  {len(gps)} records processed")
    return gps, pixels, predicts


def description_iterator(prediction_results: tuple,
                         cfg,
                         geo_map):
    """
    расшифровка результатов предсказаний
    :param prediction_results: tuple (кортеж из трех  массивов -
        (координаты исходыне (GPS) в текстовом формате строками размер - N
         координаты точек в пикселях (np.array shape (N, 2)
         предсказания классов np.array shape (N, кол-во классов))
    :param cfg:
    :param geo_map:
    :return: итератор
    """

    gps_list, pixels_arr, predicts_arr = prediction_results
    sorted_class_ids_list = np.argsort(predicts_arr, axis=1)
    class_dict = {int(k): cfg.map.class_list[k] for k in cfg.map.class_list}

    for gps, pixels, sorted_class_ids, predict in zip(gps_list,
                                                      pixels_arr,
                                                      sorted_class_ids_list,
                                                      predicts_arr):
        if geo_map.check_coord(pixels, geo_map.crop_size):
            prob = np.exp(-predict[sorted_class_ids[-1]])
            # предсказание для наиболее вероятного класса
            predicted_class1 = sorted_class_ids[-1]
            # предсказание для второго по вероятности класса
            predicted_class2 = sorted_class_ids[-2]
            if prob > cfg.threshold:
                description = f"{class_dict[predicted_class1]}"
            else:
                description = f"{class_dict[predicted_class1]}_or_{class_dict[predicted_class2]}"
        else:
            description = " ошибка координат"
            log.warning(f" ошибка координат GPS: {gps}  pixel {pixels}")

        yield PredictResultDescription(coord=pixels,
                                       gps=gps,
                                       description=description,
                                       probability=prob)


def norm_file_path(file_path, norm_path):
    if not os.path.isabs(os.path.dirname(file_path)):
        file_path = os.path.join(norm_path, file_path)
    return file_path
