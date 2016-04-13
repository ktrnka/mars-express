from __future__ import unicode_literals

import unittest

import numpy
import pandas
import sklearn_helpers
import train_test
import helpers.multivariate

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

        model = sklearn_helpers.NnRegressor(learning_rate=0.01, batch_spec=((1000, -1),), dropout=0., hidden_layer_sizes=(3,))
        model.fit(X, Y)

        Y_pred = model.predict(X)
        self.assertEqual(Y.shape, Y_pred.shape)
        error = ((Y - Y_pred) ** 2).mean().mean()
        self.assertLess(error, 1.)


class HelperTests(unittest.TestCase):
    def test_time_cross_validation_splitter(self):
        X, Y = build_data(100)

        # regular test
        splits = list(sklearn_helpers.TimeCV(X.shape[0], 4))
        self.assertListEqual([(range(0, 50), range(50, 75)), (range(0, 75), range(75, 100))], splits)

        # test with 2 buckets per test
        splits = list(sklearn_helpers.TimeCV(X.shape[0], 4, test_splits=2, balanced_tests=False))
        self.assertListEqual([(range(0, 50), range(50, 100)), (range(0, 75), range(75, 100))], splits)

        # test with no min training amount
        splits = list(sklearn_helpers.TimeCV(X.shape[0], 4, min_training=0))
        self.assertListEqual([(range(0, 25), range(25, 50)), (range(0, 50), range(50, 75)), (range(0, 75), range(75, 100))], splits)

        splits = list(sklearn_helpers.TimeCV(49125, 10))
        print [(len(s), min(s), max(s), max(s) - min(s)) for _, s in splits]
        self.assertEqual(5, len(splits))

        for train, test in splits:
            self.assertEqual(4912, len(test))

    def test_get_name(self):
        model = sklearn_helpers.NnRegressor()
        self.assertEqual("NnRegressor", sklearn_helpers.get_model_name(model))

        model = sklearn_helpers.MultivariateRegressionWrapper(sklearn_helpers.NnRegressor())
        self.assertEqual("MultivariateRegressionWrapper(NnRegressor)", sklearn_helpers.get_model_name(model))

        # test bagging
        model = helpers.multivariate.MultivariateBaggingRegressor(sklearn_helpers.NnRegressor())
        self.assertEqual("MultivariateBaggingRegressor(NnRegressor)", sklearn_helpers.get_model_name(model))

        # test random search
        model = sklearn_helpers.RandomizedSearchCV(sklearn_helpers.NnRegressor(), {"dropout": [0.4, 0.5]})
        self.assertEqual("RandomizedSearchCV(NnRegressor)", sklearn_helpers.get_model_name(model))

class UmbraTests(unittest.TestCase):
    def _make_time(self, start_time, duration_minutes=30):
        duration = pandas.Timedelta(minutes=duration_minutes)
        end_time = start_time + duration
        return {"start": start_time, "end": end_time, "duration": duration}

    def test_simple(self):
        """Test basic event-filling functionality"""
        hourly_index = pandas.DatetimeIndex(freq="1H", start=pandas.datetime(year=2016, month=4, day=1), periods=1000)

        dummy_events = [self._make_time(pandas.datetime(year=2016, month=4, day=1, hour=5, minute=50)), self._make_time(pandas.datetime(year=2016, month=4, day=1, hour=7, minute=20))]

        indicatored = train_test.get_event_series(hourly_index, dummy_events)
        self.assertEqual(1, indicatored.sum())

        minute_index = pandas.DatetimeIndex(freq="1Min", start=pandas.datetime(year=2016, month=4, day=1), periods=1000)
        indicatored = train_test.get_event_series(minute_index, dummy_events)
        self.assertEqual(60, indicatored.sum())
