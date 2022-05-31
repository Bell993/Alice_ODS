from imblearn.over_sampling import SMOTE
import pandas as pd

def smote_train(train_data, train_labels):
    """
    Увеличиваем данные ввиду несбалансированности классов
    :param train_data:
    :param train_labels:
    :return: train_data_sm, train_labels_sm
    """
    smt = SMOTE()
    train_data_sm, train_labels_sm = smt.fit_resample(train_data, train_labels)
    return train_data_sm, train_labels_sm