import os
import warnings
warnings.filterwarnings('ignore')
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

from pyriemann.estimation import ERPCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.spatialfilters import Xdawn
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold
import pickle

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

def reorder(data):
    return data.transpose(1, 0, 2)

k_folds = 10
mcc = make_scorer(matthews_corrcoef)

scores = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    # 'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
    'roc_auc': 'roc_auc',
    'balanced_accuracy': 'balanced_accuracy',
    'mcc': mcc,
}

# scores = ('accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'balanced_accuracy', mcc)
# TODO try class weight and other sampling strategies!! https://vitalflux.com/class-imbalance-class-weight-python-sklearn/#:~:text=Class%20weighting%3A%20Using%20class%20weight,represented%20relative%20to%20the%20other.
# TODO try scaling ChannellwiseScaler(StandardScaler())
clfs = {
    'LR': (
        make_pipeline(Vectorizer(), LogisticRegression(penalty='l2', solver='lbfgs', C=0.0183, random_state=42)),
        {},
    ),
    'NN': (
        make_pipeline(Vectorizer(), MLPClassifier(solver='lbfgs', activation='relu', random_state=41, alpha=54.5982)),
        {},
    ),
    'SVM': (
        make_pipeline(Vectorizer(), SVC(kernel='rbf', random_state=40, gamma=0.00001, C=54.5982)),
        {},
    ),
    'Xdawn LDA': (
        make_pipeline(Xdawn(2), Vectorizer(), LDA(shrinkage='auto', solver='eigen')),
        {},
    ),
    # 'ERPCov TS LR': (
    #     make_pipeline(ERPCovariances(estimator='oas'), TangentSpace(), LogisticRegression()),
    #     {'erpcovariances__estimator': ('lwf', 'oas')},
    # ),
    'PCA_RDA_KDE': (
        make_pipeline(FunctionTransformer(reorder), PcaRdaKdeModel(k_folds=10)),
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
    """Cross validate a record using the given classifiers and scores."""
    df = pd.DataFrame()
    for name, (clf, params) in clfs.items():
        cv = GridSearchCV(
            clf,
            params,
            scoring=scores,
            n_jobs=-1,
            refit='mcc',
            cv=k_folds,
            return_train_score=True,
            verbose=False
        )
        cv.fit(record[0], record[1])
        headers = [
            name for name in cv.cv_results_.keys()
                if name.startswith('param_') or name.startswith('mean_test_') or name.startswith('std_test_')
        ]
        results = pd.DataFrame(cv.cv_results_)[headers]
        results['classifier'] = name
        df = pd.concat((df, results), sort=False)

        # save the best model
        if session_name is not None:
            # create a models folder if it doesn't exist
            if not os.path.exists(f'{session_name}/models'):
                os.makedirs(f'{session_name}/models')
            with open(f'{session_name}/models/model_{name}.pkl', 'wb') as f:
                pickle.dump(cv.best_estimator_, f)
    return df.reindex(sorted(df.columns), axis=1)
