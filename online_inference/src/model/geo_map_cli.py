#!/usr/bin/python
# -*- coding: UTF-8 -*-
""""
версия 6.0.1
программа анализа геоданных по изображению генплана города.
на входе файл массив с GPS координатами в формате
 [55.86605046256052, 49.07751363142922]
[55.85844989210669, 49.097790215192205]
[55.86076205652522, 49.1011128167036]
одна строка - 1 запрос
 на выходе файл с описанием

[55.79883508185922, 49.105875912272566];14404;12460;Специализированная зона размещения объектов торговли...
[55.80330934948255, 49.33272207144011];25250;12114;Зона мест погребения

пример запуска в режиме отладки. на вход подается файл запросов error_test.csv  . изображения районов координат
 сохраняются в каталог /home/sokolov/debug
/src/model/geo_map_cli.py input=data/queries/error_test.csv debug=True debug_image_dir=/home/sokolov/debug

"""

import os.path
import logging

import numpy as np
import torch
from PIL import Image
import hydra

from utils.common_utils import norm_file_path
from utils.transformers import GeoTransformer, GpsToPixelTransformer
from utils.datasets import QueryDataset
from utils.data_structures import Config, GeoImage
from utils.common_utils import batch_predict_area_class, description_iterator

log = logging.getLogger(__name__)
MAX_FILE_NAME_LEN = 150


def write_predict_w_description(output_file_name: str,
                                prediction_results: tuple,
                                cfg,
                                geo_map):
    """
    сохраниет результат работы в файл
    :param output_file_name:
    :param prediction_results:
    :param cfg:
    :param geo_map: объект - изображение карты
    :return:
    """
    cnt = 0
    log.info(f" Try to save to   {output_file_name} file")
    with open(output_file_name, "wt", encoding="utf-8") as fout:
        for point_w_description in description_iterator(prediction_results, cfg, geo_map):

            fout.write(f"{point_w_description.gps};"
                       f"{point_w_description.coord[0]};{point_w_description.coord[1]};"
                       f"{point_w_description.description}\n")
            cnt += 1
            if cfg.debug:
                file_name = f"img_" \
                            f"{point_w_description.gps.replace('.', '_')}" \
                            f"_x{point_w_description.coord[0]}-y{point_w_description.coord[1]}" \
                            f"_{round(100 * point_w_description.probability)}_" \
                            f"_{point_w_description.description}" \

                #  обрезаем имя файла
                if len(file_name) > MAX_FILE_NAME_LEN:
                    file_name = file_name[:MAX_FILE_NAME_LEN]

                file_name += "_.jpg"
                file_name = os.path.join(cfg.debug_image_dir, file_name)
                geo_map.save_debug_image(file_name,
                                         point_w_description.coord,
                                         crop_size=cfg.debug_crop_size)
        log.info(f" saved {cnt} records")


@hydra.main(config_path="conf",
            config_name="config.yaml")
def main(cfg: Config) -> None:
    log.info(f"Start")
    root_path = os.path.normpath(os.path.join(hydra.utils.get_original_cwd(), cfg.path_to_root))
    input_file = norm_file_path(cfg.input, root_path)
    output_file = norm_file_path(cfg.output, root_path)

    Image.MAX_IMAGE_PIXELS = max(Image.MAX_IMAGE_PIXELS, cfg.map.max_image_pixels)

    geo_map = GeoImage(os.path.join(root_path, cfg.map.map_image), cfg.map.crop_size)

    gps_to_pixel = GpsToPixelTransformer(np.array(cfg.map.gps_coord),
                                         np.array(cfg.map.pixel_coord))

    to_tensor_transformer = GeoTransformer(geo_map)
    log.info(f"Loading  model {cfg.map.model_pkl}")
    model = torch.load(os.path.join(root_path, cfg.map.model_pkl))
    model.eval()

    query_dataset = QueryDataset(input_file,
                                 gps_to_pixel.transform,
                                 to_tensor_transformer
                                 )
    query_data_loader = torch.utils.data.DataLoader(query_dataset,
                                                    batch_size=cfg.batch_size,
                                                    # num_workers=1,
                                                    shuffle=False,
                                                    drop_last=False)

    prediction_results = batch_predict_area_class(model, query_data_loader, cfg)

    write_predict_w_description(output_file, prediction_results, cfg, geo_map)
    log.info("End processing data")


if __name__ == "__main__":
    main()
