import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.estimator_checks import check_estimator


class AlphaTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        baseline_start_s: float,
        baseline_duration_s: float,
        response_start_s: float,
        response_duration_s: float,
        sample_rate_hz: int,
        *,
        z_score_per_trial=False
    ):
        # Times are relative to the start of each trial
        # Region to compute mean and stdev for z-scoring
        self.baseline_start_s = baseline_start_s
        self.baseline_duration_s = baseline_duration_s

        # Region of interest, where user response occurs
        self.response_start_s = response_start_s
        self.response_duration_s = response_duration_s
        self.sample_rate_hz = sample_rate_hz
        self.z_score_per_trial = z_score_per_trial

    def fit(self, X, y=None):
        return self

    def transform(self, X, do_slice=True):
        baseline_begin = int(self.baseline_start_s * self.sample_rate_hz)
        baseline_end = baseline_begin + int(self.baseline_duration_s * self.sample_rate_hz)
        baseline_data = X[..., baseline_begin:baseline_end]

        q5 = np.percentile(baseline_data, q=5, axis=(0, 1))
        q95 = np.percentile(baseline_data, q=95, axis=(0, 1))
        baseline_data = np.clip(baseline_data, q5, q95)

        if self.z_score_per_trial:
            means = baseline_data.mean(axis=2, keepdims=True)  # (1000 *4) means
            stdevs = baseline_data.std(axis=2, keepdims=True)  # (1000 *4) stdevs
        else:
            means = baseline_data.mean(axis=(0, 2), keepdims=True)  # (4) means
            stdevs = baseline_data.std(axis=(0, 2), keepdims=True)  # (4) stdevs

        response_begin = int(self.response_start_s * self.sample_rate_hz)
        response_end = response_begin + int(self.response_duration_s * self.sample_rate_hz)
        if do_slice:
            response_data = X[..., response_begin:response_end]
        else:
            response_data = X

        response_data = (response_data - means) / stdevs
        return response_data


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    check_estimator(AlphaTransformer())

    x, y = make_classification()
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    pipeline = make_pipeline(AlphaTransformer(), LogisticRegression())
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
