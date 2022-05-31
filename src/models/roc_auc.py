from sklearn.metrics import roc_auc_score


def roc_auc(val_labels, predict, output_score):
    """
    Загрузка параметров из файла yaml
    :param val_labels:
    :param predict:
    :param output_score:
    :return:
    """
    roc = roc_auc_score(val_labels, predict)
    file = open(output_score, "w+")
    content = str(roc)
    file.write(content)
    file.close()
