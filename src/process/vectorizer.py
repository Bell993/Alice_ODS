from sklearn.feature_extraction.text import CountVectorizer


def vector(input_path):
    """
    Векторизирует датасет
    :param input_path:
    :return: X: векторизированные данные датасета
    """
    cv = CountVectorizer(ngram_range=(1, 3), max_features=50000)
    with open(input_path) as inp_train_file:
        X = cv.fit_transform(inp_train_file)
    return X
