#!/usr/bin/python
# -*- coding: UTF-8 -*-
""""
набор Pytest тестов для тестирования restfull версии приложения geo_map

"""
import pytest

from fastapi.testclient import TestClient

from app_fast_api import app

client = TestClient(app)


def test_read_root():
    response = client.get("/", headers={"X-Token": "coneofsilence"})
    assert response.status_code == 200
    answer = response.content.decode(response.encoding)
    etalon_answer = "\"Приложение для определения типа земельного участка по его GPS координатам и изображению генплана." \
                    " Конфигурация для города Казань \""
    assert answer == etalon_answer


def test_404_error():
    response = client.get("/abracadabra", headers={"X-Token": "coneofsilence"})
    assert response.status_code == 404
    answer = response.content.decode(response.encoding)
    etalon_answer = '{"detail":"Not Found"}'
    assert answer == etalon_answer


def test_read_status():
    with TestClient(app) as client:
        response = client.get("/status", headers={"X-Token": "coneofsilence"})
        assert response.status_code == 200
        answer = response.content.decode(response.encoding)
        etalon_answer = '"   Model is loaded:  True   GPS transformer is loaded: True <br>   Features transformer is loaded: True <br>   Map is loaded: True"'
        assert answer == etalon_answer


def test_predict():
    with TestClient(app) as client:
        response = client.get("/predict/[55.80330934948255,%2049.33272207144011]", headers={"X-Token": "coneofsilence"})
        assert response.status_code == 200
        answer = response.content.decode(response.encoding)
        etalon_answer = '"[\\"PredictResultDescription(gps=\'[55.80330934948255, 49.33272207144011]\', coord=array([25250, 12114]), probability=1.0000129, description=\'Зона мест погребения\')\\"]"'
        assert answer == etalon_answer

        response = client.get("/predict/[55.79883508185922,%2049.105875912272566]", headers={"X-Token": "coneofsilence"})
        assert response.status_code == 200
        answer = response.content.decode(response.encoding)
        etalon_answer = '"[\\"PredictResultDescription(gps=\'[55.79883508185922, 49.105875912272566]\', coord=array([14404, 12460]), probability=1.0000265, description=\'Специализированная зона размещения объектов торговли, образования, здравоохранения, культуры, спорта\')\\"]"'
        assert answer == etalon_answer
