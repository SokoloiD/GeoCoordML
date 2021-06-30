#!/usr/bin/python
# -*- coding: UTF-8 -*-
""""
версия 6.2.1
Restfull верксия анализа геоданных по изображению генплана города.

"""
import os.path
import logging
import numpy as np
from pydantic import BaseModel
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException
import json

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from omegaconf import OmegaConf, DictConfig

from dataclasses import dataclass

from utils.transformers import GeoTransformer, GpsToPixelTransformer
from utils.data_structures import Config, GeoImage
from utils.common_utils import description_iterator


@dataclass
class AppConfig:
    features_transformer: transforms
    to_pixel_transformer: transforms
    model: nn.Module
    geo_map: GeoImage
    cfg: DictConfig


app = FastAPI(title="Geo coord ")
log = logging.getLogger(__name__)
app_config = AppConfig(None, None, None, None, None)
test_var = ""


@app.on_event("startup")
async def set_up():
    global app_config
    cfg = OmegaConf.load("conf/config_web_app.yaml")
    map_cfg = OmegaConf.load("conf/map/kazan_cfg.yaml")
    cfg.map = map_cfg
    log.info(f"Start")

    global  test_var
    test_var = os.getcwd()

    root_path = os.path.normpath(os.path.join(os.getcwd(), cfg.path_to_root))
    Image.MAX_IMAGE_PIXELS = max(Image.MAX_IMAGE_PIXELS, cfg.map.max_image_pixels)
    app_config.geo_map = GeoImage(os.path.join(root_path, cfg.map.map_image), cfg.map.crop_size)
    app_config.to_pixel_transformer = GpsToPixelTransformer(np.array(cfg.map.gps_coord),
                                                            np.array(cfg.map.pixel_coord))
    app_config.cfg = cfg
    app_config.features_transformer = GeoTransformer(app_config.geo_map)

    log.info(f"Loading  model {cfg.map.model_pkl}")
    app_config.model = torch.load(os.path.join(root_path, cfg.map.model_pkl))
    app_config.model.eval()


@app.get("/")
def root() -> str:
    return "Приложение для определения типа земельного участка по его GPS координатам и изображению генплана." \
           " Конфигурация для города Казань "


def convert_and_check_coord(gps_coord_str: str):
    err_code = "ok"
    status = True
    gps_coord = None
    try:
        gps_coord = np.array(list(map(float, gps_coord_str.strip('[ ]\n').split(", "))))
        if gps_coord.shape != (2,):
            status = False
            err_code = f"wrong shape for {gps_coord}"
        if not (-180. <= gps_coord[0] <= 180 and
                -90 < -gps_coord[1] <= 180):
            status = False
            err_code = f"wrong value for {gps_coord}"

    except Exception:
        err_code = f"wrong GPS format: {gps_coord_str}. use [55.7988, 49.105]"
        status = False

    return status, gps_coord, err_code


@app.get("/predict/{gps_coord_str}")
async def predict(gps_coord_str: str):
    status, gps_coord, err_description = convert_and_check_coord(gps_coord_str)
    if not status:
        raise HTTPException(status_code=400, detail=err_description)
    pix_coors = app_config.to_pixel_transformer(gps_coord)

    pict_tensor = app_config.features_transformer(pix_coors)
    pict_tensor = torch.unsqueeze(pict_tensor, 0)
    predict = app_config.model(pict_tensor).cpu().detach().numpy()
    iterator = description_iterator(([gps_coord_str], [pix_coors], predict),
                                    app_config.cfg, app_config.geo_map)
    return_value = [str(p) for p in iterator]
    json_compatible_item_data = json.dumps(return_value,ensure_ascii=False).encode('utf8')
    return json_compatible_item_data


@app.get("/status")
def get_status() -> bool:
    global app_config
    state = f"   Model is loaded:  {app_config.model is not None}" \
            f"   GPS transformer is loaded: {app_config.to_pixel_transformer is not None} <br>" \
            f"   Features transformer is loaded: {app_config.features_transformer is not None} <br>" \
            f"   Map is loaded: {app_config.geo_map is not None}"
    return state


if __name__ == "__main__":
    uvicorn.run("app_fast_api:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
