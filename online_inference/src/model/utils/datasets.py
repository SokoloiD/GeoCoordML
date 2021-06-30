#!/usr/bin/python
# -*- coding: UTF-8 -*-
""""
версия 1.0.1
классы для работы со входными данными.
для тренировки модели - набор маршрутав

"""

import logging
import numpy as np
import pandas as pd
from torch.utils import data

from utils.data_structures import GeoImage

log = logging.getLogger(__name__)


class QueryDataset(data.Dataset):
    """
    Датасет для обработки запросов в формате gps координат точек
    """

    def __init__(self, query_file_name, gps_to_pixel_transformer, pixel_to_tensor_transformer):
        super(QueryDataset, self).__init__()
        self.gps = []
        self.gps_to_pixel_transformer = gps_to_pixel_transformer
        self.to_tensor_transformer = pixel_to_tensor_transformer
        log.info(f"Open query file {query_file_name}")
        with open(query_file_name, "rt") as fin:
            for line in fin:
                self.gps.append(line.strip())
            log.info(f" {len(self.gps)} lines read")

    def __len__(self):
        return len(self.gps)

    def __getitem__(self, idx):
        gps_coord = list(map(float, self.gps[idx].strip('[ ]\n').split(", ")))
        pix_coord = self.gps_to_pixel_transformer(gps_coord)
        return {"gps": self.gps[idx],
                "pixels": pix_coord,
                "tensor": self.to_tensor_transformer(pix_coord)
                }


class WalkLinesToDataset(data.Dataset):
    """
    Датасет для обучения модели. На преобразует маршруты по замельным участкам в последовательность изображений карты

    """
    def __init__(self, image_file_name, walk_file_name, crop_size=16, walk_step=5, transforms=None):
        super(WalkLinesToDataset, self).__init__()
        self.transforms = transforms
        self.rectangle_size = crop_size
        self.map_image = GeoImage(image_file_name, crop_size)
        self.targets = []
        x_coords = []
        y_coords = []
        walks_df = pd.read_csv(walk_file_name, sep="\t", header=0)
        walks_df = walks_df.fillna(0).astype(np.int)
        for _, row in walks_df.iterrows():
            class_num = row[0]
            walk_points = row[1:].to_numpy().reshape(-1, 2)
            from_x, from_y = walk_points[0, 0], walk_points[0, 1]
            for to_x, to_y in walk_points[1:]:
                if to_x == 0 or to_y == 0:
                    break
                d_x = to_x - from_x
                d_y = to_y - from_y

                distance = (d_x ** 2 + d_y ** 2) ** 0.5
                steps = np.arange(0, distance, walk_step)
                size = steps.shape[0]

                x_steps = from_x + steps * d_x / distance
                y_steps = from_y + steps * d_y / distance
                self.targets.append(np.full((size,), class_num, dtype=np.int64))
                x_coords.append(x_steps.astype(np.int))
                y_coords.append(y_steps.astype(np.int))
                from_x, from_y = to_x, to_y

        self.targets = np.concatenate(self.targets)
        x_coords = np.concatenate(x_coords)
        y_coords = np.concatenate(y_coords)
        self.coords = np.stack([x_coords, y_coords], axis=1)

        assert len(self.targets) == self.coords.shape[0]

    def __getitem__(self, idx):
        sample = {"targets": self.targets[idx]}
        points = {"coord": self.coords[idx]}
        if self.transforms is not None:
            points = self.transforms(points)
        sample["image"] = self.map_image.get_rectangle(points["coord"],
                                                       self.rectangle_size)

        sample["realcoord"] = points["coord"]

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.targets)
