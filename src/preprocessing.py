"""Training-only preprocessing and feature selection transformers."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each feature to quantiles learned from the fitted sample."""

    def __init__(self, lower=0.005, upper=0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        frame = self._as_frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.lower_bounds_ = frame.quantile(self.lower)
        self.upper_bounds_ = frame.quantile(self.upper)
        return self

    def transform(self, X):
        frame = self._as_frame(X, self.feature_names_in_)
        return frame.clip(self.lower_bounds_, self.upper_bounds_, axis=1)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

    @staticmethod
    def _as_frame(X, columns=None):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X, columns=columns)


class CorrelationMISelector(BaseEstimator, TransformerMixin):
    """Rank by mutual information, then remove correlated lower-ranked features."""

    def __init__(self, max_features=30, corr_threshold=0.90, random_state=42):
        self.max_features = max_features
        self.corr_threshold = corr_threshold
        self.random_state = random_state

    def fit(self, X, y):
        frame = self._as_frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        scores = mutual_info_classif(
            frame, np.asarray(y), discrete_features=False,
            random_state=self.random_state
        )
        self.mi_scores_ = pd.Series(scores, index=frame.columns).sort_values(ascending=False)
        corr = frame.corr(method="spearman").abs()
        selected = []
        for feature in self.mi_scores_.index:
            if all(corr.loc[feature, kept] < self.corr_threshold for kept in selected):
                selected.append(feature)
            if len(selected) >= self.max_features:
                break
        self.selected_features_ = selected
        self.dropped_features_ = [c for c in frame.columns if c not in selected]
        return self

    def transform(self, X):
        frame = self._as_frame(X, self.feature_names_in_)
        return frame.loc[:, self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)

    @staticmethod
    def _as_frame(X, columns=None):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X, columns=columns)

