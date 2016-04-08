from __future__ import unicode_literals
import sys
import argparse
import unittest
# import nose.util
import numpy

import pandas
from src import sklearn_helpers
from src.train_test import fill_events


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def main():
    args = parse_args()

if __name__ == "__main__":
    sys.exit(main())

def _test_multivariate_regression(model, X, Y):
    model.fit(X, Y)
    return (model.predict(Y) - Y).mean().mean()


def build_data(n):
    X = numpy.asarray(range(n))

    X = numpy.vstack((X, X + 1, X + 2, X + 3)).transpose()

    return X[:,:2], X[:,2:]


class SklearnHelperTests(unittest.TestCase):
    def test_nn_regression(self):
        X, Y = build_data(100)
        self.assertListEqual([100, 2], list(X.shape))
        self.assertListEqual([100, 2], list(Y.shape))

        model = sklearn_helpers.NnRegressor(learning_rate=0.01, batch_spec=((1000, -1),), dropout=0., hidden_layer_sizes=(3,))
        model.fit(X, Y)

        Y_pred = model.predict(X)
        self.assertEqual(Y.shape, Y_pred.shape)
        error = ((Y - Y_pred) ** 2).mean().mean()
        self.assertLess(error, 1.)

class UmbraTests(unittest.TestCase):
    def _make_time(self, start_time, duration_minutes=30):
        duration = pandas.Timedelta(minutes=duration_minutes)
        end_time = start_time + duration
        return {"start": start_time, "end": end_time, "duration": duration}

    def test_single(self):
        """Test short 30 min events"""
        df = pandas.DataFrame(index=pandas.DatetimeIndex(freq="1H", start=pandas.datetime(year=2016, month=4, day=1), periods=1000))

        dummy_events = [self._make_time(pandas.datetime(year=2016, month=4, day=1, hour=5, minute=50)), self._make_time(pandas.datetime(year=2016, month=4, day=1, hour=7, minute=20))]

        fill_events(df, dummy_events, "IN_UMBRA")
        self.assertEqual(1, df["IN_UMBRA"].sum())

    def test_multi(self):
        """Test cases that cross multiple time ranges"""
        df = pandas.DataFrame(index=pandas.DatetimeIndex(freq="1H", start=pandas.datetime(year=2016, month=4, day=1), periods=1000))

        dummy_events = [self._make_time(pandas.datetime(year=2016, month=4, day=1, hour=5, minute=50)), self._make_time(pandas.datetime(year=2016, month=4, day=1, hour=7, minute=20), duration_minutes=101)]

        fill_events(df, dummy_events, "IN_UMBRA")
        self.assertEqual(3, df["IN_UMBRA"].sum())

    def test_durations(self):
        df = pandas.DataFrame(index=pandas.DatetimeIndex(freq="1H", start=pandas.datetime(year=2016, month=4, day=1), periods=1000))

        dummy_events = [self._make_time(pandas.datetime(year=2016, month=4, day=1, hour=5, minute=50))]

        # first event should add 10 minutes to one and 20 to the other

        fill_events(df, dummy_events, "UMBRA_DURATION", add_duration=True)
        self.assertEqual(30 * 60, df["UMBRA_DURATION"].sum())
