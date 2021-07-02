#!/usr/bin/python
# -*- coding: UTF-8 -*-
""""
версия 6.1.0
sort_images.py утилита для проверки правильности классификации.
На вход получат так же как и train.py файл с маршрутами, проходящих по территории одного класса.
Маршрут задается набором координат точек. Формат - csv
class x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 x6 y6 x7 y7 x8 y8 x9 y9 x10 y10
0 16227 13054 16300 13017 16286 12929 16117 12904 16105 13024
0 15830 12909
1 19043 11043 18991 11038 19064 10836 19090 10836 19150 10962 19140 10990
подсчитывает количество неправильных классификация и разбивкой по парам (target-predit) .
Так же может в целях отладки сохранять изображение окрестностей точек, для которой проводилась классификация

"""

import os.path
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split

from PIL import Image
from omegaconf import DictConfig
import hydra
from utils.datasets import WalkLinesToDataset
from utils.common_utils import norm_file_path
from utils.transformers import ValidateTransformer
from utils.data_structures import PredictResult
import tqdm
from collections import defaultdict

log = logging.getLogger(__name__)


def save_images(results,
                walk_lines,
                save_wrong_classified=True,
                top_cnt: int = 10,
                cfg=None,
                save_dir="."):
    image_count = 0

    for res in results:
        if (res.target != res.predict) == save_wrong_classified:
            image_count += 1
            image_file_name = f"img_groundtrue_{res.target}-pred_{res.predict}_{round(100 * np.exp(-res.score))}" \
                              f"_pecent_x{res.coord[0]}_y{res.coord[1]}.jpg"
            walk_lines.map_image.save_debug_image(os.path.join(save_dir, image_file_name),
                                                  res.coord,
                                                  crop_size=cfg.debug_crop_size)

            if image_count == top_cnt:
                break
    log.info(f" {image_count} images saved to {save_dir} dir. "
             f"Image size is {cfg.debug_crop_size}x{cfg.debug_crop_size} pix."
             f"Wrong classified = {save_wrong_classified}")


@hydra.main(config_name='sort_images_config.yaml',
            config_path="conf", )
def main(cfg: DictConfig) -> None:
    log.info("Start scoring")
    root_path = os.path.normpath(os.path.join(hydra.utils.get_original_cwd(), cfg.path_to_root))
    Image.MAX_IMAGE_PIXELS = cfg.map.max_image_pixels
    train_transforms = ValidateTransformer(cfg.deviation)
    Image.MAX_IMAGE_PIXELS = max(Image.MAX_IMAGE_PIXELS, cfg.map.max_image_pixels)
    log.info(f"Try to load path database {os.path.join(root_path, cfg.walks_file)}")
    walk_lines = WalkLinesToDataset(walk_file_name=os.path.join(root_path, cfg.walks_file),
                                    image_file_name=os.path.join(root_path, cfg.map.map_image),
                                    transforms=train_transforms,
                                    crop_size=cfg.map.crop_size,
                                    walk_step=cfg.walk_step)
    dataset_len = len(walk_lines)
    log.info(f"Dataset len {dataset_len}")

    val_data_loader = torch.utils.data.DataLoader(walk_lines, batch_size=1, shuffle=False)

    model = torch.load(os.path.join(root_path, cfg.map.model_pkl))
    loss_fn = nn.NLLLoss()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    log.info(f"Working on {device}")
    model.eval()
    model.to(device)
    results = []

    for point in tqdm.tqdm(val_data_loader, total=len(val_data_loader), desc="scoring...", position=0, leave=True):
        predict = model(point["image"]).cpu()
        score = loss_fn(predict, point["targets"])
        predicted_result = PredictResult()
        predicted_result.target = point["targets"].detach().numpy()[0]
        predicted_result.predict = np.argmax(predict[0].detach().numpy())
        predicted_result.coord = point["realcoord"].detach().numpy()[0]
        predicted_result.score = score.item()
        results.append(predicted_result)

    results.sort(key=lambda x: 10 - x.score)

    save_images(results,
                walk_lines,
                save_wrong_classified=True,
                top_cnt=cfg.save_miclassified_images,
                cfg=cfg,
                save_dir=norm_file_path(cfg.debug_image_dir, root_path))

    save_images(results,
                walk_lines,
                save_wrong_classified=False,
                top_cnt=cfg.save_miclassified_images,
                cfg=cfg,
                save_dir=norm_file_path(cfg.debug_image_dir, root_path))

    mismatch_cnt = 0
    mismatch_map = defaultdict(int)
    for res in results:
        if res.target != res.predict:
            mismatch_cnt += 1
            mismatch_map[(res.target, res.predict)] += 1
    log.info(f" mismatch_cnt {mismatch_cnt} mismatch ratio {mismatch_cnt / len(results)}")
    for k, v in mismatch_map.items():
        log.info(f"mismatch {k}   {v}")


if __name__ == "__main__":
    main()
