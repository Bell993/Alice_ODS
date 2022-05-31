from sklearn.linear_model import LogisticRegression
from src.models.roc_auc import roc_auc
def logit(train_data,train_labels,val_data,val_labels,output_score,**params):
    """
    Модель логистической регресии. Параметры принимаются из файла, результат ввиде метрики roc_auc записывается в файл
    :param train_data:
    :param train_labels:
    :param val_data:
    :param val_labels:
    :param output_score:
    :param params:
    :return:
    """

    logistic = LogisticRegression(**params)
    logistic.fit(train_data, train_labels)
    y_pred = logistic.predict(val_data)
    roc_auc(val_labels,y_pred,output_score)