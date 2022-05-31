import pandas as pd

def clean(input_path,output_path):
    """
    Заполнение пропусков значениями 0 и сохранение в output
    :param input_path:
    :param output_path:
    :return:
    """
    df = pd.read_csv(input_path, index_col='session_id')
    df = df.fillna(0).astype('int')
    df.to_csv(output_path,sep=' ', index=None, header=None)