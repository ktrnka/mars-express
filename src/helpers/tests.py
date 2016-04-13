from __future__ import unicode_literals

import unittest

import helpers.neural
import helpers.sk
import numpy
import sklearn.pipeline


class HelperTests(unittest.TestCase):
    def test_time_cross_validation_splitter(self):
        X, Y = build_data(100)

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


def _test_multivariate_regression(model, X, Y):
    model.fit(X, Y)
    return (model.predict(Y) - Y).mean().mean()


def build_data(n):
    X = numpy.asarray(range(n))

    X = numpy.vstack((X, X + 1, X + 2, X + 3)).transpose()

    return X[:,:2], X[:,2:]


class ModelTests(unittest.TestCase):
    def test_nn_regression_model(self):
        X, Y = build_data(100)
        self.assertListEqual([100, 2], list(X.shape))
        self.assertListEqual([100, 2], list(Y.shape))

        model = helpers.neural.NnRegressor(learning_rate=0.01, num_epochs=1000, dropout=0., hidden_layer_sizes=(3,))
        model.fit(X, Y)

        Y_pred = model.predict(X)
        self.assertEqual(Y.shape, Y_pred.shape)
        error = ((Y - Y_pred) ** 2).mean().mean()
        self.assertLess(error, 1.)