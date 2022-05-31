import pandas as pd
from sklearn.model_selection import train_test_split


def split(X_value, df_to_target: str, test_size: float):
    """
    Сплитит векторизированные данные X и данные таргета
    :param X_value:sparce matrix
    :param df_to_target:pandas series
    :param test_size:
    :return: train_data, val_data, train_labels, val_labels
    """
    X = X_value
    y = pd.read_csv(df_to_target)
    y = y['target'].values
    train_data, val_data, train_labels, val_labels = train_test_split(X, y, random_state=50)

    return train_data, val_data, train_labels, val_labels
