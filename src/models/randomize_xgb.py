from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from src.params.load_params import save_yaml


def search_params(train_data, train_labels, output_param, **params):
    """
    Поиск гипермараметров модели XGB путем RandomizedSearchCV. Метрика roc_auc, лучшие параметры записываются в yaml файл
    :param train_data:
    :param train_labels:
    :param output_param:
    :param params:
    :return:
    """
    model = XGBClassifier(n_jobs=2, eval_metric='auc')
    randomsearch = RandomizedSearchCV(model, param_distributions=params, n_iter=5, scoring='roc_auc', cv=5)
    randomsearch.fit(train_data, train_labels)
    best_param = randomsearch.best_params_
    print(randomsearch.best_score_)

    save_yaml(best_param, output_param)

    #запись параметра в csv при необходимости
    # file = open(output_param, "w+")
    # content = str(best_param)
    # file.write(content)
    # file.close()
