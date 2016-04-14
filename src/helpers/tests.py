from __future__ import unicode_literals

import unittest

import helpers.neural
import helpers.sk
import numpy
import sklearn.pipeline
import sklearn.linear_model
import sklearn.metrics


class HelperTests(unittest.TestCase):
    def test_time_cross_validation_splitter(self):
        X, Y = _build_data(100)

        # regular test
        splits = list(helpers.sk.TimeCV(X.shape[0], 4))
        self.assertListEqual([(range(0, 50), range(50, 75)), (range(0, 75), range(75, 100))], splits)

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

    return X[:,:2], X[:,2:]

def _build_periodic_data(n):
    X = numpy.asarray(range(n))

    X = numpy.vstack((X, X + 1, numpy.sin(X / 50))).transpose()
    return X[:,:2], X[:,2:]

def _build_identity(n):
    return numpy.asarray(range(n))[:, numpy.newaxis], numpy.asarray(range(n))[:, numpy.newaxis]


class ModelTests(unittest.TestCase):
    def test_nn_identity(self):
        X, Y = _build_identity(100)
        model = helpers.neural.NnRegressor(learning_rate=0.5, input_noise=0.00001, num_epochs=200, loss="mae", dropout=None, hidden_units=5, early_stopping=True, verbose=1)
        model.fit(X, Y)

        # This test is incredibly frustrating
        self.assertAlmostEqual(0, sklearn.metrics.mean_absolute_error(Y, model.predict(X)))

    def test_nn_regression_model(self):
        X, Y = _build_data(100)

        model = helpers.neural.NnRegressor(learning_rate=0.01, num_epochs=1000, input_noise=0.01, dropout=0., hidden_layer_sizes=(3,))
        model.fit(X, Y)

        Y_pred = model.predict(X)
        self.assertEqual(Y.shape, Y_pred.shape)
        error = ((Y - Y_pred) ** 2).mean().mean()
        self.assertLess(error, 1.)

    def test_rnn(self):
        X, Y = _build_periodic_data(1000)

        print X.shape, Y.shape

        baseline_model = sklearn.linear_model.LinearRegression().fit(X, Y)

        Y_pred = baseline_model.predict(X)
        self.assertEqual(Y.shape, Y_pred.shape)
        baseline_error = sklearn.metrics.mean_squared_error(Y, Y_pred)
        self.assertAlmostEqual(0.5, baseline_error, places=2)

        # test non-RNN to see how it does

        # add an extra axis for time, which is ignored
        X = X[:, numpy.newaxis, :]

        model = helpers.neural.NnRegressor(learning_rate=0.01, num_epochs=50, batch_size=10, rnn_spec=helpers.neural.RnnSpec(5), verbose=2)
        model.fit(X, Y)

        Y_pred = model.predict(X)
        self.assertEqual(Y.shape, Y_pred.shape)
        error = ((Y - Y_pred) ** 2).mean().mean()
        self.assertLess(error, 0.1)

        self.assertLessEqual(error, baseline_error)

        # TODO
        # Test identity function

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

        model = helpers.sk.MultivariateBaggingRegressor(base_estimator=sklearn.linear_model.LinearRegression(), max_samples=0.8, max_features=0.6)
        model.fit(X, Y)

        Y_pred = model.predict(X)
        self.assertEqual(Y.shape, Y_pred.shape)
        model_error = sklearn.metrics.mean_squared_error(Y, Y_pred)
        self.assertLess(model_error, 1.)

        # test that it's an improvement within some epsilon
        self.assertLessEqual(model_error, baseline_error + 1e-6)
