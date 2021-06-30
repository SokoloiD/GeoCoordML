#!/usr/bin/python
# -*- coding: UTF-8 -*-
""""
набор Скрипт для тестирования   restfull версии приложения geo_map
пример использования
python script_test_web.py http://127.0.0.1:8000


"""
import sys
import requests


def test_read_root(url: str) -> bool:
    response = requests.get(url + "/", headers={"X-Token": "coneofsilence"})
    assert response.status_code == 200
    answer = response.content.decode("utf-8")  # response.encoding)
    etalon_answer = "\"Приложение для определения типа земельного участка по его GPS координатам и изображению генплана." \
                    " Конфигурация для города Казань \""
    return answer == etalon_answer


def test_404_error(url: str):
    response = requests.get(url + "/abracadabra", headers={"X-Token": "coneofsilence"})
    assert_1 = response.status_code == 404
    answer = response.content.decode("utf-8")  # response.encoding)
    etalon_answer = '{"detail":"Not Found"}'
    assert_2 = answer == etalon_answer

    return assert_1 and assert_2


def test_read_status(url: str):
    response = requests.get(url + "/status", headers={"X-Token": "coneofsilence"})
    assert_1 = response.status_code == 200
    answer = response.content.decode("utf-8")  # response.encoding)
    etalon_answer = '"   Model is loaded:  True   GPS transformer is loaded: True <br>   Features transformer is loaded: True <br>   Map is loaded: True"'
    assert_2 = answer == etalon_answer

    return assert_1 and assert_2


def test_predict(url: str):
    response = requests.get(url + "/predict/[55.80330934948255,%2049.33272207144011]",
                            headers={"X-Token": "coneofsilence"})
    assert_1 = response.status_code == 200
    answer = response.content.decode("utf-8")  # response.encoding)
    etalon_answer = '"[\\"PredictResultDescription(gps=\'[55.80330934948255, 49.33272207144011]\', coord=array([25250, 12114]), probability=1.0000129, description=\'Зона мест погребения\')\\"]"'
    assert_2 = answer == etalon_answer

    response = requests.get(url + "/predict/[55.79883508185922,%2049.105875912272566]",
                            headers={"X-Token": "coneofsilence"})
    assert_3 = response.status_code == 200
    answer = response.content.decode("utf-8")  # response.encoding)
    etalon_answer = '"[\\"PredictResultDescription(gps=\'[55.79883508185922, 49.105875912272566]\', coord=array([14404, 12460]), probability=1.0000265, description=\'Специализированная зона размещения объектов торговли, образования, здравоохранения, культуры, спорта\')\\"]"'
    assert_4 = answer == etalon_answer

    return assert_1 and assert_2 and assert_3 and assert_4


def main():
    test_list = [test_read_root, test_read_status, test_404_error, test_predict]
    url = sys.argv[1]
    pass_count = 0
    for test in test_list:
        if test(url):
            pass_count += 1
    print(f" Выполнено {len(test_list)} тестов. Пройдено {pass_count}. ")


if __name__ == "__main__":
    main()
