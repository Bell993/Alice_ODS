import yaml
import numpy as np
from yaml.loader import SafeLoader




def save_yaml(params, output):
    """
    Сохранение гиперпараметров модели в yaml файл
    :param params:
    :param output:
    :return:
    """
    with open(output, 'w') as fr:
        yaml.dump(params, fr)


def load_yaml(input):
    """
    Загрузка гиперпараметров модели из yaml
    :param input:
    :return:
    """
    with open(input, 'r') as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data


