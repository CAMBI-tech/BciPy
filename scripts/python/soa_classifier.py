from sklearn.model_selection import GridSearchCV, PredefinedSplit
import numpy as np
import pandas as pd

from mne.decoding import Vectorizer, CSP
from bcipy.signal.model import PcaRdaKdeModel, RdaKdeModel

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score, make_scorer, roc_auc_score
from sklearn.svm import SVC

from pyriemann.estimation import ERPCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.spatialfilters import Xdawn

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

def reorder(data):
    return data.transpose(1, 0, 2)

k_folds = 2

scores = ('accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'balanced_accuracy')

clfs = {
    'LR': (
        make_pipeline(Vectorizer(), LogisticRegression()),
        {'logisticregression__C': np.exp(np.linspace(-4, 4, 5))},
    ),
    'LDA': (
        make_pipeline(Vectorizer(), LDA(shrinkage='auto', solver='eigen')),
        {},
    ),
    'SVM': (
        make_pipeline(Vectorizer(), SVC()),
        {'svc__C': np.exp(np.linspace(-4, 4, 5)), 'svc__kernel': ('linear', 'rbf')},
    ),
    'Xdawn LDA': (
        make_pipeline(Xdawn(2), Vectorizer(), LDA(shrinkage='auto', solver='eigen')),
        {},
    ),
    'ERPCov TS LR': (
        make_pipeline(ERPCovariances(estimator='oas'), TangentSpace(), LogisticRegression()),
        {'erpcovariances__estimator': ('lwf', 'oas')},
    ),
    'PCA_RDA_KDE': (
        make_pipeline(FunctionTransformer(reorder), PcaRdaKdeModel(k_folds=k_folds)),
        {},
    ),
    'RDA_KDE': (
        make_pipeline(FunctionTransformer(reorder), RdaKdeModel(k_folds=k_folds)),
        {},
    ),
        
}

def crossvalidate_record(record, clfs=clfs, scores=scores):
    """Cross validate a record using the given classifiers and scores."""
    df = pd.DataFrame()
    for name, (clf, params) in clfs.items():
        cv = GridSearchCV(
            clf,
            params,
            scoring=scores,
            n_jobs=-1,
            refit=False,
            cv=k_folds,
        )
        cv.fit(record[0], record[1])
        headers = [
            name for name in cv.cv_results_.keys()
                if name.startswith('param_') or name.startswith('mean_test_') or name.startswith('std_test_')
        ]
        results = pd.DataFrame(cv.cv_results_)[headers]
        results['classifier'] = name
        df = pd.concat((df, results), sort=False)
    return df.reindex(sorted(df.columns), axis=1)
