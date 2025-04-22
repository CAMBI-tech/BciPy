import os
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import numpy as np
import pandas as pd

from mne.decoding import Vectorizer, CSP
from bcipy.signal.model import PcaRdaKdeModel, RdaKdeModel

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import balanced_accuracy_score, make_scorer, roc_auc_score, matthews_corrcoef
from sklearn.svm import SVC

from pyriemann.estimation import ERPCovariances, Covariances, XdawnCovariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from pyriemann.spatialfilters import Xdawn
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold
import pickle

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import parallel_backend
from sklearn.exceptions import ConvergenceWarning
import warnings

def reorder(data):
    return data.transpose(1, 0, 2)

k_folds = 10
mcc = make_scorer(matthews_corrcoef)


scores = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    # 'f1': 'f1',
    # 'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
    'roc_auc': 'roc_auc',
    'balanced_accuracy': 'balanced_accuracy',
    'mcc': mcc,
}

# scores = ('accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'balanced_accuracy', mcc)
clfs = {
    'LR': (
        make_pipeline(Vectorizer(), LogisticRegression(penalty='l2', solver='lbfgs', C=0.0183, random_state=42, max_iter=1000)),
        {},
    ),
    'NN': (
        make_pipeline(Vectorizer(), MLPClassifier(solver='lbfgs', activation='relu', random_state=41, alpha=54.5982, max_iter=1000)),
        {},
    ),
    'SVM': (
        make_pipeline(Vectorizer(), SVC(kernel='rbf', random_state=40, gamma=0.00001, C=54.5982)),
        {},
    ),
    'TS-LDA': (
        make_pipeline(XdawnCovariances(3), TangentSpace(metric="riemann"), LDA(shrinkage='auto', solver='eigen')),
        {},
    ),
    'MDM': (
        make_pipeline(XdawnCovariances(3), MDM()),
        {},
    ),
    'PCA_RDA_KDE': (
        make_pipeline(FunctionTransformer(reorder), PcaRdaKdeModel(k_folds=k_folds)),
        {},
    ),
    'Random Model': (
        make_pipeline(DummyClassifier(strategy='uniform', random_state=39)),
        {}
    )
    # 'RDA_KDE': (
    #     make_pipeline(FunctionTransformer(reorder), RdaKdeModel(k_folds=k_folds)),
    #     {},
    # ),
}

def crossvalidate_record(record, clfs=clfs, scores=scores, session_name=None):
    """Cross validate a record using the given classifiers and scores.
    
    record: tuple of (X, y) where X is the data and y is the labels
    clfs: dict of classifiers to use for cross validation
    scores: dict of scores to use for cross validation
    session_name: str, name of the session to save the models
    """
    df = pd.DataFrame()
    for name, (clf, params) in clfs.items():
        cv = GridSearchCV(
            clf,
            params,
            scoring=scores,
            n_jobs=-1,
            refit='balanced_accuracy',
            cv=k_folds,
            return_train_score=False,
            verbose=1
        )
        cv.fit(record[0], record[1])
        headers = [
            name for name in cv.cv_results_.keys()
                if name.startswith('param_') or name.startswith('mean_test_') or name.startswith('std_test_')
        ]
        results = pd.DataFrame(cv.cv_results_)[headers]
        results['classifier'] = name
        df = pd.concat((df, results), sort=False)
        
        # get the best score for the classifier
        best_score = cv.best_score_
        # save the best model
        if session_name is not None:
            # create a models folder if it doesn't exist
            if not os.path.exists(f'{session_name}/models'):
                os.makedirs(f'{session_name}/models')
            with open(f'{session_name}/models/model_{name}_{best_score}.pkl', 'wb') as f:
                pickle.dump(cv.best_estimator_, f)
    return df.reindex(sorted(df.columns), axis=1)