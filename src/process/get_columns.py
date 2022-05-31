import pandas as pd
import click


def get_sites(input_path:str,output_path:str):
    """
    Загрузка файла и получение данных сессий с названием сайтов
    :param input_path:
    :param output_path:
    :return: сохранение файла в csv
    """

    df = pd.read_csv(input_path,index_col='session_id')
    sites = ['site' + str(i) for i in range(1, 11)]
    df = df[sites]
    df.to_csv(output_path,index='session_id')


