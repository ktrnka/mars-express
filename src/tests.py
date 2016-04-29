from __future__ import unicode_literals

import unittest

import pandas

import helpers.features
import train_test


class UmbraTests(unittest.TestCase):
    def _make_time(self, start_time, duration_minutes=30):
        duration = pandas.Timedelta(minutes=duration_minutes)
        end_time = start_time + duration
        return {"start": start_time, "end": end_time, "duration": duration}

    def test_simple(self):
        """Test basic event-filling functionality"""
        hourly_index = pandas.DatetimeIndex(freq="1H", start=pandas.datetime(year=2016, month=4, day=1), periods=1000)

        dummy_events = [self._make_time(pandas.datetime(year=2016, month=4, day=1, hour=5, minute=50)), self._make_time(pandas.datetime(year=2016, month=4, day=1, hour=7, minute=20))]

        indicatored = helpers.features.get_event_series(hourly_index, dummy_events)
        self.assertEqual(1, indicatored.sum())

        minute_index = pandas.DatetimeIndex(freq="1Min", start=pandas.datetime(year=2016, month=4, day=1), periods=1000)
        indicatored = helpers.features.get_event_series(minute_index, dummy_events)
        self.assertEqual(60, indicatored.sum())
