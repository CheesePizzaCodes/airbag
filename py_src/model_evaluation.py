import os
from typing import Tuple
import multiprocess as mp
from itertools import product

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_validate

from file_io import load_preprocessed_data
from file_io import append_to_json_file


class ClassifierFamily:
    def __init__(self, verbose=False):
        self.logistic_regression = LogisticRegression(random_state=0,
                                                      max_iter=10000,
                                                      class_weight='balanced', verbose=verbose)

        self.random_forest = RandomForestClassifier(n_estimators=100, random_state=0, verbose=verbose)

        # self.svm = SVC(kernel='linear', class_weight='balanced', random_state=0, verbose=verbose, shrinking=False)

        self.naive_bayes = GaussianNB()

        self.knn = KNeighborsClassifier(n_neighbors=5)

        self.model_names = [name for name in dir(self) if '__' not in name]

    def get_model__(self, model_name: str):
        return getattr(self, model_name)

    def get_models__(self):
        """
        Private method. Not to be called outside of __init__
        """
        return [getattr(self, model_name) for model_name in self.model_names]


def run_cross_validation(classifier_name: str,
                         data: Tuple[np.ndarray, np.ndarray],
                         dataset_name: str,
                         n_splits: int = 5):
    """

    @param dataset_name:
    @param classifier_name:
    @param data:
    @param n_splits:
    @return:
    """
    classifier = ClassifierFamily(verbose=True).get_model__(classifier_name)

    X, y = data

    # Create a KFold object with 5 folds and no shuffling
    k_fold = KFold(n_splits=n_splits, shuffle=True)

    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, average='weighted'),
               'recall': make_scorer(recall_score, average='weighted'),
               'f1': make_scorer(f1_score, average='weighted')}

    scores = cross_validate(classifier, X, y, cv=k_fold, scoring=scoring, n_jobs=7)
    report = {metric: f"{scores.mean():.2f} (+/- {scores.std() * 2:.2f})" for metric, scores in scores.items()}

    report.update({'classifier': str(classifier),
                   'dataset_name': str(dataset_name)})
    print(report)

    try:
        append_to_json_file('./results/results6.json', report)
    except Exception as e:
        print(e)


def test_models():
    """
    tests all models
    """
    classifiers_family = ClassifierFamily(verbose=True)
    classifiers = classifiers_family.get_models__()

    for classifier in classifiers:
        print(f'Classifier: {classifier}')
        data = load_preprocessed_data(window_size_overlap_tuple=(20, 0.5))
        run_cross_validation(classifier, data, n_splits=2)
        ...


if __name__ == '__main__':
    test_models()
