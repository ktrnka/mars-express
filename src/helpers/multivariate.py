import unittest

import numpy
import sklearn
import sklearn.base
import sklearn.utils.random
import sklearn.metrics
import sklearn.linear_model
import logging

def _convert_scale(target_value, max_value):
    if target_value <= 1:
        return int(max_value * target_value)

    assert target_value <= max_value

    return target_value


class SubspaceWrapper(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(self, base_estimator=None, max_samples=1.0, max_features=1.0):
        self.base_estimator = base_estimator
        self.max_samples = max_samples
        self.max_features = max_features

        self.logger_ = logging.getLogger("SubspaceWrapper")
        self.cols_ = None
        self.estimator_ = None

    def fit(self, X, Y):
        rows = sklearn.utils.random.sample_without_replacement(X.shape[0], _convert_scale(self.max_samples, X.shape[0]))
        self.cols_ = sklearn.utils.random.sample_without_replacement(X.shape[1], _convert_scale(self.max_features, X.shape[1]))

        self.logger_.debug("Rows for %f: %s", self.max_samples, rows)
        self.logger_.debug("Cols for %f: %s", self.max_features, self.cols_)

        self.estimator_ = sklearn.base.clone(self.base_estimator).fit(X[rows][:, self.cols_], Y[rows])
        return self

    def predict(self, X):
        return self.estimator_.predict(X[:, self.cols_])


class MultivariateBaggingRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features

        self.logger = logging.getLogger("MultivariateBaggingRegressor")
        self.estimators_ = None

    def _get_estimator(self):
        return SubspaceWrapper(self.base_estimator, self.max_samples, self.max_features)

    def fit(self, X, Y):
        assert len(Y.shape) == 2

        self.estimators_ = [self._get_estimator().fit(X, Y) for _ in xrange(self.n_estimators)]
        return self

    def predict(self, X):
        result = numpy.dstack([estimator.predict(X) for estimator in self.estimators_])

        assert len(result.shape) == 3
        assert result.shape[0] == X.shape[0]

        result = result.mean(axis=2)
        assert len(result.shape) == 2

        return result


def _build_data(n):
    X = numpy.asarray(range(n))

    X = numpy.vstack((X, X + 1, X + 2, X + 3)).transpose()

    return X[:, :2], X[:, 2:]


class ModelTests(unittest.TestCase):
    def test_build_data(self):
        X, Y = _build_data(100)
        self.assertListEqual([100, 2], list(X.shape))
        self.assertListEqual([100, 2], list(Y.shape))

    def test_model(self):
        X, Y = _build_data(100)

        # test basic linear regression
        baseline_model = sklearn.linear_model.LinearRegression().fit(X, Y)

        Y_pred = baseline_model.predict(X)
        self.assertEqual(Y.shape, Y_pred.shape)
        baseline_error = sklearn.metrics.mean_squared_error(Y, Y_pred)
        self.assertLess(baseline_error, 1.)

        model = MultivariateBaggingRegressor(base_estimator=sklearn.linear_model.LinearRegression(), max_samples=0.8, max_features=0.6)
        model.fit(X, Y)

        Y_pred = model.predict(X)
        self.assertEqual(Y.shape, Y_pred.shape)
        model_error = sklearn.metrics.mean_squared_error(Y, Y_pred)
        self.assertLess(model_error, 1.)

        # test that it's an improvement within some epsilon
        self.assertLessEqual(model_error, baseline_error + 1e-6)
