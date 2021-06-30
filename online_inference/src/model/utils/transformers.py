#!/usr/bin/python
# -*- coding: UTF-8 -*-
""""
версия 1.0.1
трансформеры для обработки с GPS  координат  точек, рандомизации коррдинат пиксельных координат точек,
 обработки избражений карты
"""
import torch
from torchvision import transforms
import numpy as np

from utils.data_structures import GeoImage

TORCH_PRETRAINED_MEAN = [0.485, 0.456, 0.406]
TORCH_PRETRAINED_STD = [0.229, 0.224, 0.225]


class GeoTransformer:
    def __init__(self, image_map: GeoImage):
        self.transformer = transforms.Compose([
            image_map.get_rectangle,
            transforms.ToTensor(),
            transforms.Normalize(mean=TORCH_PRETRAINED_MEAN, std=TORCH_PRETRAINED_STD),
        ])

    def __call__(self, sample):
        return self.transformer(sample)


class TrainTransformer:
    def __init__(self, deviation: int):
        self.transformer = transforms.Compose([
            TransformByKeys(RandomizeCoords(deviation=deviation), ("coord",)),
            TransformByKeys(transforms.RandomHorizontalFlip(), ("image",)),
            TransformByKeys(transforms.RandomVerticalFlip(), ("image",)),

            TransformByKeys(transforms.ToTensor(), ("image",)),
            TransformByKeys(transforms.Normalize(mean=TORCH_PRETRAINED_MEAN, std=TORCH_PRETRAINED_STD), ("image",)),
        ])

    def __call__(self, sample):
        return self.transformer(sample)


class ValidateTransformer:
    def __init__(self, deviation: int):
        self.transformer = transforms.Compose([
            TransformByKeys(RandomizeCoords(deviation=deviation), ("coord",)),
            TransformByKeys(transforms.ToTensor(), ("image",)),
            TransformByKeys(transforms.Normalize(mean=TORCH_PRETRAINED_MEAN, std=TORCH_PRETRAINED_STD), ("image",)),
        ])

    def __call__(self, sample):
        return self.transformer(sample)


class TransformByKeys:
    def __init__(self, transform, names):
        self.transform = transform
        self.names = set(names)

    def __call__(self, sample):
        for name in self.names:
            if name in sample:
                sample[name] = self.transform(sample[name])

        return sample


class RandomizeCoords:
    def __init__(self, deviation=5):
        self.deviation = deviation

    def __call__(self, coord):
        rand_offset = torch.randint(- self.deviation,
                                    self.deviation,
                                    (2,)).numpy()
        return coord + rand_offset


class GpsToPixelTransformer:
    '''
    трансформер для преобразования коррдинат GPS в пиксели

    '''

    def __init__(self, gps_coord=None, pixels_coord=None):
        # матрица линейных преобразований координат
        self.rotate_matrix = None
        if gps_coord is not None and pixels_coord is not None:
            self.fit(gps_coord, pixels_coord)

    def fit(self, gps_coord, pixels_coord):
        """
        формирование матрицы преобразований
        :param pixels_coord: координанты 3х точек pixels numpy.array  shape (3, 2)
        :param gps_coord: координанты 3х точек GPS numpy.array  shape (3, 2)   dtype=np.float64

        :return:
        """

        gps = np.ones((3, 3), dtype=np.float64)
        gps[:, [0, 1]] = gps_coord

        pixels = np.ones((3, 3), dtype=np.float64)
        pixels[:, [0, 1]] = pixels_coord

        self.rotate_matrix = np.linalg.inv(gps) @ pixels

    def transform(self, gps_coord: np.array):
        """
        преобразование координат. на входе  GPS numpy.array  shape ( N, 2)
        :param gps_coord: GPS numpy.array  shape (2,)

        :return:  pixels numpy.array  shape (2,)
        """

        coord = np.ones((3,), dtype=np.float64)

        coord[[0, 1]] = gps_coord
        transformed = coord.reshape((1, 3)) @ self.rotate_matrix
        return transformed[0, [0, 1]].astype(np.int)

    def __call__(self, gps_coord: np.array):
        return self.transform(gps_coord)