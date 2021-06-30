#!/usr/bin/python
# -*- coding: UTF-8 -*-
""""
версия 1.0.1
набор классов для конфигурации приложения


"""
from dataclasses import dataclass
from omegaconf import DictConfig, MISSING
from PIL import Image, ImageDraw
import numpy as np
import logging

log = logging.getLogger(__name__)


class GeoImage:
    """
    Класс для храниения изображения карты и параметров выборки из нее
     фрагментов изображений для анализа, сохранения части изображения по координатам

    """

    def __init__(self, file_name: str, crop_size: int):
        self.crop_size = crop_size
        self.image = None
        self.load_image(file_name)

    def load_image(self, file_name: str):
        log.info(f"Try to load map image {file_name}")
        self.image = Image.open(file_name)

    def check_coord(self, coord, size: int) -> bool:
        """
        проверка, попадает ли квадрат с цетром в coord полностью в изображение карты
        :param coord: np.array (x,y)
        :param size:
        :return: True если квадрат полностью вписывается
        """

        half_size = size // 2

        if half_size < coord[0] < self.image.size[0] - half_size and \
                half_size < coord[1] < self.image.size[1] - half_size:
            return True
        return False

    def get_rectangle(self, coord: np.array, size: int = -1):
        """
        Возращает изображение квадратного размера с центром в координатах coord
        :param coord: np.array [x, y] центр изображения
        :param size: размер стороны
        :return: картинка PIL image
        если координаты выходят за пределы карты возвращает левый прямоугольник

        """
        if size == -1:
            size = self.crop_size
        half_size = size // 2
        if not self.check_coord(coord, size):
            log.warning(f"ошибка координат {coord}")
            coord = np.array([half_size + 1, half_size + 1])

        rect = np.concatenate([coord - half_size, coord + half_size]).astype(int)
        img = self.image.crop(rect)
        return img.convert('RGB')

    def save_debug_image(self, file_name: str,
                         coord: np.array,
                         crop_size: int
                         ):
        """
        сохренение отладочного изображения в файл. На отладочном изображении крестом отображается выбранная точка и
        квадратом область для анализа
        :param file_name:
        :param coord: np.array(x,y)
        :param crop_size: сторона квадрата сохраняемого изображения

        :return: None

        если коодинаты не вписываются в изображение - ничего не делает
        """

        half_size = crop_size // 2
        half_crop = self.crop_size // 2
        if not self.check_coord(coord, crop_size):
            log.warning(f"Ошибка координат :{coord} для файла {file_name}")
            return

        im_crop = self.get_rectangle(coord, crop_size)

        draw = ImageDraw.Draw(im_crop)
        draw.line((half_size - half_crop, half_size - half_crop,
                   half_size + half_crop, half_size - half_crop))
        draw.line((half_size - half_crop, half_size + half_crop,
                   half_size + half_crop, half_size + half_crop))
        draw.line((half_size - half_crop, half_size - half_crop,
                   half_size - half_crop, half_size + half_crop))
        draw.line((half_size + half_crop, half_size - half_crop,
                   half_size + half_crop, half_size + half_crop))

        # рисуем крест
        draw.line((half_size, 0, half_size, crop_size))
        draw.line((0, half_size, crop_size, half_size))

        im_crop.save(file_name)


@dataclass
class MapConfig(DictConfig):
    """
    структура для конфигурации карты местности
    """
    map_image: str = MISSING  # файл изображения
    model_pkl: str = MISSING  # тренированная модель
    max_image_pixels: int = MISSING  # количество пикселей в изображении
    gps_coord: list = MISSING  # кооодинаты GPS трех точек [[],[],[]] 3, 2
    pixel_coord: list = MISSING  # кооодинаты пикселей  трех точек [[],[],[]] 3, 2
    crop_size: int = MISSING  # размер изображения, подаваемого в модель
    class_list: dict = MISSING  # перечень названий классов, на которы модель кообучена


@dataclass
class Config(DictConfig):
    """
    класс для хранения конфиураыии основной программы
    """
    input: str = "input.csv"
    output: str = "output.csv"
    debug: bool = False
    debug_crop_size: int = 128
    debug_image_dir: str = "c:/temp"
    batch_size: int = 512
    threshold: float = 0.95
    # путь к корневому каталогу ПРОГРАММЫ
    path_to_root: str = MISSING
    map: MapConfig = MISSING


@dataclass
class PredictResult:
    coord: tuple = (0, 0)
    target: int = 0
    predict: int = 0
    score: float = 0


@dataclass
class PredictResultDescription:
    gps: str = ""
    coord: np.array = np.array([0, 0])
    probability: float = 0
    description: str = ""

