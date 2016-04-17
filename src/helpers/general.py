from __future__ import unicode_literals

import datetime
import inspect
import logging
import os

import time

import numpy

import helpers.sk


def number_string(number, singular_unit, plural_unit, format_string="{} {}"):
    return format_string.format(number, singular_unit if number == 1 else plural_unit)


class Timed(object):
    """Decorator for timing how long a function takes"""
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        retval = self.func(*args, **kwargs)
        elapsed = time.time() - start_time

        hours, seconds = divmod(elapsed, 60 * 60)
        minutes = seconds / 60.
        time_string = number_string(minutes, "minute", "minutes", format_string="{:.1f} {}")
        if hours:
            time_string = ", ".join((number_string(hours, "hour", "hours"), time_string))

        print "{} took {}".format(self.func.__name__, time_string)

        return retval


def prepare_time_matrix(X, time_steps=5, fill_value=None):
    if time_steps < 1:
        raise ValueError("time_steps must be 1 or more")

    assert isinstance(X, numpy.ndarray)
    time_shifts = [X]
    time_shifts.extend(numpy.roll(X, t, axis=0) for t in range(1, time_steps))
    time_shifts = reversed(time_shifts)

    X_time = numpy.dstack(time_shifts)
    X_time = X_time.swapaxes(1, 2)

    if fill_value is not None:
        for t in range(time_steps):
            missing_steps = time_steps - t
            X_time[t, :missing_steps-1, :] = fill_value

    return X_time


def _with_extra(filename, extra_info):
    base, ext = os.path.splitext(filename)
    return "".join([base, ".", extra_info, ext])


def with_num_features(filename, X):
    return _with_extra(filename, "{}_features".format(X.shape[1]))


def with_model_name(filename, model):
    return _with_extra(filename, helpers.sk.get_model_name(model, format="{}_{}"))


def with_date(filename):
    return _with_extra(filename, datetime.datetime.now().strftime("%m_%d"))


def get_function_logger(num_calls_ago=1):
    _, file_name, _, function_name, _, _ = inspect.stack()[num_calls_ago]
    if file_name:
        file_name = os.path.basename(file_name)
    return logging.getLogger("{}:{}".format(file_name, function_name))

def get_class_logger(obj):
    return logging.getLogger(type(obj).__name__)
