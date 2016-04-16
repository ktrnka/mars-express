from __future__ import unicode_literals

import unittest

import math

import helpers.neural
import helpers.sk
import numpy
import sklearn.pipeline
import sklearn.linear_model
import sklearn.metrics
import sklearn.dummy

class HelperTests(unittest.TestCase):
    def test_time_cross_validation_splitter(self):
        X, Y = _build_data(100)

        # regular test
        # TODO: I refactored this so that it'd fail in the same way as RandomizedSearchCV
        splits = helpers.sk.TimeCV(X.shape[0], 4)
        self.assertSequenceEqual([(range(0, 50), range(50, 75)), (range(0, 75), range(75, 100))], splits)
        self.assertEqual(2, len(splits))

        # test with 2 buckets per test
        splits = list(helpers.sk.TimeCV(X.shape[0], 4, test_splits=2, balanced_tests=False))
        self.assertListEqual([(range(0, 50), range(50, 100)), (range(0, 75), range(75, 100))], splits)

        # test with no min training amount
        splits = list(helpers.sk.TimeCV(X.shape[0], 4, min_training=0))
        self.assertListEqual([(range(0, 25), range(25, 50)), (range(0, 50), range(50, 75)), (range(0, 75), range(75, 100))], splits)

        splits = list(helpers.sk.TimeCV(49125, 10))
        print [(len(s), min(s), max(s), max(s) - min(s)) for _, s in splits]
        self.assertEqual(5, len(splits))

        for train, test in splits:
            self.assertEqual(4912, len(test))

    def test_get_name(self):
        model = helpers.neural.NnRegressor()
        self.assertEqual("Nn", helpers.sk.get_model_name(model))

        model = helpers.neural.NnRegressor()
        self.assertEqual("NnRegressor", helpers.sk.get_model_name(model, remove=None))

        model = helpers.sk.MultivariateRegressionWrapper(helpers.neural.NnRegressor())
        self.assertEqual("MultivariateWrapper(Nn)", helpers.sk.get_model_name(model))

        # test bagging
        model = helpers.sk.MultivariateBaggingRegressor(helpers.neural.NnRegressor())
        self.assertEqual("MultivariateBagging(Nn)", helpers.sk.get_model_name(model))

        # test random search
        model = helpers.sk.RandomizedSearchCV(helpers.neural.NnRegressor(), {"dropout": [0.4, 0.5]})
        self.assertEqual("RandomizedSearchCV(Nn)", helpers.sk.get_model_name(model))

        # test a pipeline
        pipe = sklearn.pipeline.Pipeline([("nn", helpers.neural.NnRegressor())])
        self.assertEqual("Pipeline(Nn)", helpers.sk.get_model_name(pipe))

        # test format on pipeline
        self.assertEqual("Pipeline_Nn", helpers.sk.get_model_name(pipe, format="{}_{}"))


def _test_multivariate_regression(model, X, Y):
    model.fit(X, Y)
    return (model.predict(Y) - Y).mean().mean()


def _build_data(n):
    X = numpy.asarray(range(n))

    X = numpy.vstack((X, X + 1, X + 2, X + 3)).transpose()

    return X[:, :2], X[:, 2:]


def _build_periodic_data(n, period=50.):
    X = numpy.asarray(range(n), dtype=numpy.float32)

    X = numpy.vstack((X, numpy.cos(2. * math.pi * X / period))).transpose()
    return X[:, 0].reshape((-1, 1)), X[:, 1].reshape((-1, 1))

def moving_average(a, n=3) :
    """From http://stackoverflow.com/a/14314054/1492373"""
    ret = numpy.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def _build_summation_data(n, lag=4):
    X = (numpy.random.rand(n, 1) > 0.5).astype(numpy.int16)

    Y = moving_average(X, lag).reshape(-1, 1)

    return X[lag-1:,:], Y


def _build_identity(n):
    X = numpy.random.rand(n, 1)
    return X, X


class ModelTests(unittest.TestCase):
    def test_nn_identity(self):
        X, Y = _build_identity(100)

        baseline_model = sklearn.dummy.DummyRegressor("mean").fit(X, Y)
        baseline_error = sklearn.metrics.mean_squared_error(Y, baseline_model.predict(X))

        nn = helpers.neural.NnRegressor(learning_rate=0.05, num_epochs=200, hidden_units=5, verbose=1)
        nn.fit(X, Y)
        nn_error = sklearn.metrics.mean_squared_error(Y, nn.predict(X))

        # should fit better than baseline
        self.assertLess(nn_error, baseline_error)
        self.assertLess(nn_error, baseline_error / 100)

        # should be able to fit the training data completely (but doesn't, depending on the data)
        self.assertAlmostEqual(0, nn_error, places=4)

    def test_nn_regression_model(self):
        # TODO: Replace this with Boston dataset or something
        X, Y = _build_data(100)

        model = helpers.neural.NnRegressor(learning_rate=0.01, num_epochs=1000, hidden_units=3)
        model.fit(X, Y)

        Y_pred = model.predict(X)
        self.assertEqual(Y.shape, Y_pred.shape)
        error = ((Y - Y_pred) ** 2).mean().mean()
        self.assertLess(error, 1.)

    def test_rnn(self):
        X, Y = _build_summation_data(1000, lag=4)

        # baseline linear regression
        baseline_model = sklearn.linear_model.LinearRegression().fit(X, Y)

        baseline_predictions = baseline_model.predict(X)
        self.assertEqual(Y.shape, baseline_predictions.shape)
        baseline_error = sklearn.metrics.mean_squared_error(Y, baseline_predictions)
        self.assertLess(baseline_error, 0.1)

        # test non-RNN
        model = helpers.neural.NnRegressor(activation="tanh", batch_size=50, num_epochs=100, verbose=0, early_stopping=True)
        model.fit(X, Y)
        mlp_predictions = model.predict(X)
        self.assertEqual(Y.shape, mlp_predictions.shape)
        mlp_error = ((Y - mlp_predictions) ** 2).mean().mean()
        self.assertLess(mlp_error, baseline_error * 1.2)

        # test RNN
        model = helpers.neural.RnnRegressor(num_epochs=200, batch_size=50, num_units=50, time_steps=5, early_stopping=True)
        model.fit(X, Y)
        rnn_predictions = model.predict(X)

        self.assertEqual(Y.shape, rnn_predictions.shape)
        error = ((Y - rnn_predictions) ** 2).mean().mean()

        print "RNN error", error

        # should be more than 10x better
        self.assertLessEqual(error, mlp_error / 10)

    def test_build_data(self):
        X, Y = _build_data(100)
        self.assertListEqual([100, 2], list(X.shape))
        self.assertListEqual([100, 2], list(Y.shape))

    def test_bagging(self):
        X, Y = _build_data(100)

        import sklearn.linear_model
        import sklearn.metrics

        # test basic linear regression
        baseline_model = sklearn.linear_model.LinearRegression().fit(X, Y)

        Y_pred = baseline_model.predict(X)
        self.assertEqual(Y.shape, Y_pred.shape)
        baseline_error = sklearn.metrics.mean_squared_error(Y, Y_pred)
        self.assertLess(baseline_error, 1.)

        model = helpers.sk.MultivariateBaggingRegressor(base_estimator=sklearn.linear_model.LinearRegression(),
                                                        max_samples=0.8, max_features=0.6)
        model.fit(X, Y)

        Y_pred = model.predict(X)
        self.assertEqual(Y.shape, Y_pred.shape)
        model_error = sklearn.metrics.mean_squared_error(Y, Y_pred)
        self.assertLess(model_error, 1.)

        # test that it's an improvement within some epsilon
        self.assertLessEqual(model_error, baseline_error + 1e-6)
