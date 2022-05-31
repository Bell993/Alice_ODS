from xgboost import XGBClassifier
from src.models.roc_auc import roc_auc


def xgb(train_data, train_labels, val_data, val_labels, output_score, **params):
    """
    Модель XGBClassifier принимает параметры из yaml файла, обучает и сохраняет результаты метрики roc_auc в файл
    :param train_data:
    :param train_labels:
    :param val_data:
    :param val_labels:
    :param output_score:
    :param params:
    :return:
    """
    model = XGBClassifier(n_jobs=4, eval_metric='auc', **params)
    model.fit(train_data, train_labels)
    y_pred = model.predict(val_data)
    roc_auc(val_labels, y_pred, output_score)
