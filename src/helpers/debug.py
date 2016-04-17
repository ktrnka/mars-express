"""
Data is weird? Model is doing weird shit? This module helps but only if you have beer.
"""

from __future__ import unicode_literals
import unittest
import numpy

import sklearn.linear_model
import sklearn.cross_validation
from sklearn.metrics import mean_absolute_error
import sklearn.ensemble


def explain_prediction(y_true, y_pred, x, model, eps=1e-4, param_eps=None, max_eps=20.):
    assert isinstance(x, numpy.ndarray)
    if param_eps is None:
        param_eps = numpy.ones_like(x) * eps

    # squared error
    base_error = (y_true - y_pred) ** 2

    # build a perturbation matrix of size the number of elements
    pos_shifted = stack_all_perturbations(x, 0, perturb_elements=param_eps)
    assert len(pos_shifted.shape) == 2

    perturbed_predictions = model.predict(pos_shifted)

    # elementwise squared error
    perturbed_error = (y_true - perturbed_predictions) ** 2

    d_error = (perturbed_error - base_error).flatten()
    d_x = x * param_eps / 2.

    input_gradients = d_error / d_x

    # if any gradients are zero it could be that our eps is too small
    if any(grad == 0 for grad in input_gradients):
        scale = (input_gradients == 0).astype(numpy.float32) * 10 + 1

        wider_eps = numpy.minimum(scale * param_eps, numpy.ones_like(scale) * max_eps)

        if not numpy.array_equal(param_eps, wider_eps):
            return explain_prediction(y_true, y_pred, x, model, param_eps=wider_eps, max_eps=max_eps)

    return input_gradients


def explain_blame(feature_blame):
    for i, derivative in enumerate(feature_blame):
        print "To get desired prediction, the model wants you to adjust feature {} by {}".format(i, -derivative)


def stack_all_perturbations(x, perturb_all, perturb_elements=None, dtype=numpy.float32):
    """If x is a vector, this generates a square matrix and we perturb the matrix along the diagonal"""
    perturbations = numpy.tile(x.reshape(1, -1), (x.shape[0], 1)).astype(dtype)

    assert perturbations.shape[0] == perturbations.shape[1]

    perturb_matrix = numpy.ones_like(perturbations)
    if perturb_all:
        perturb_matrix += numpy.identity(perturb_matrix.shape[0]) * perturb_all
    if perturb_elements is not None:
        perturb_matrix += numpy.diag(perturb_elements)

    return perturbations * perturb_matrix


class ExplanationTests(unittest.TestCase):
    @staticmethod
    def _make_data(n):
        base = numpy.random.rand(n, 1).flatten()
        X = numpy.asarray([numpy.cumsum(numpy.ones_like(base)), numpy.ones_like(base)]).transpose()
        Y = (base + X[:, 0]).reshape(-1, 1)

        return X, Y

    def test_perturbation(self):
        row = numpy.asarray([1, 2, 3])

        perturbations = stack_all_perturbations(row, .1)

        for i in xrange(len(row)):
            for j in xrange(len(row)):
                if i == j:
                    self.assertAlmostEqual(1.1 * row[i], perturbations[i, j], places=5)
                else:
                    self.assertEqual(row[j], perturbations[i, j])

    def test_explain_linear_regression(self):
        X, Y = self._make_data(100)
        X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size=0.2)

        # train a model, no intercept so that it's easier to diagnose
        model = sklearn.linear_model.LinearRegression(fit_intercept=False).fit(X_train, Y_train)
        predictions = model.predict(X_test)

        # check that it has a semi-ok fit
        self.assertLess(mean_absolute_error(Y_test, predictions), 10)

        # now diagnose all examples - it should be weighting the first feature the most
        for i in xrange(Y_test.shape[0]):
            input_gradient = explain_prediction(Y_test[i], predictions[i], X_test[i], model)
            self.assertLess(abs(input_gradient[1]), abs(input_gradient[0]))

    def test_explain_random_forest(self):
        X, Y = self._make_data(100)
        X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size=0.2)

        model = sklearn.ensemble.RandomForestRegressor(5, max_depth=5, random_state=4).fit(X_train, Y_train)
        predictions = model.predict(X_test)

        # check that it has a semi-ok fit
        self.assertLess(mean_absolute_error(Y_test, predictions), 10)

        # now diagnose all examples - it should be weighting the first feature the most
        for i in xrange(Y_test.shape[0]):
            input_gradient = explain_prediction(Y_test[i], predictions[i], X_test[i], model)
            self.assertLess(abs(input_gradient[1]), abs(input_gradient[0]))
