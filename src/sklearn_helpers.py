from __future__ import unicode_literals

import math
import numbers
import sys
import argparse
from operator import itemgetter

import collections

import numpy
import pandas
import scipy.optimize
import sklearn
import sklearn.grid_search
import sklearn.linear_model
import keras.constraints
import keras.layers.noise
import keras.optimizers
import keras.callbacks
import keras.models
import scipy.stats

def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    args = parse_args()


if __name__ == "__main__":
    sys.exit(main())


class TimeSeriesRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(self):
        self.params_ = None

    @staticmethod
    def get_time_offset(X, base_time):
        assert isinstance(X, pandas.DataFrame)
        assert isinstance(X.index, pandas.DatetimeIndex)

        x = (X.index - base_time).astype(numpy.int64) / 10 ** 6
        return x

    @staticmethod
    def _time_function(t, base, linear_amplitude, log_amplitude, periodic_amplitude, periodic_period, periodic_offset):
        y = base + linear_amplitude * t + log_amplitude * numpy.log(t) + periodic_amplitude * numpy.sin(periodic_offset + t / (2 * math.pi * periodic_period))
        return y

    @staticmethod
    def _simple_time_function(t, base, linear_amplitude, periodic_amplitude, periodic_offset, periodic_period):
        y = base + linear_amplitude * t + periodic_amplitude * numpy.sin(periodic_offset + t / (2 * math.pi * periodic_period))
        return y

    @staticmethod
    def _get_time_function_defaults(y):
        time_range = y[-1] - y[0]
        numeric_range = y.max() - y.min()

        return y[0], time_range, time_range, numeric_range, 687 * 24 * 60 * 60 * 1000., 1

    def fit(self, x, y):
        assert len(x.shape) == 1
        optimal_params, covariance = scipy.optimize.curve_fit(TimeSeriesRegressor._simple_time_function, x, y, xtol=0.09)
        self.params_ = optimal_params

        return self

    def predict(self, x):
        assert len(x.shape) == 1
        return numpy.asarray([self._simple_time_function(t, *self.params_) for t in x])


class MultivariateRegressionWrapper(sklearn.base.BaseEstimator):
    """
    Wrap a univariate regression model to support multivariate regression.
    Tweaked from http://stats.stackexchange.com/a/153892
    """
    def __init__(self, estimator):
        self.estimator = estimator
        self.estimators_ = None

    def fit(self, X, Y):
        if isinstance(Y, pandas.DataFrame):
            Y = Y.values

        assert len(Y.shape) == 2
        self.estimators_ = [sklearn.base.clone(self.estimator).fit(X, Y[:, i]) for i in xrange(Y.shape[1])]
        return self

    def predict(self, X):
        result = numpy.hstack([estimator.predict(X)[:, numpy.newaxis] for estimator in self.estimators_])

        assert result.shape[0] == X.shape[0]
        assert result.shape[1] == len(self.estimators_)

        return result

    def get_best_param_distributions(self):
        """Get distributions over grid search parameters for the best models on each output"""

        params = collections.defaultdict(list)
        for estimator in self.estimators_:
            for k, v in estimator.best_params_.iteritems():
                params[k].append(v)

        return {k: numpy.asarray(v) for k, v in params.iteritems()}

    def print_best_params(self):
        print "Best hyperparameters for grid search inside of multivariate regression"

        for name, dist in self.get_best_param_distributions().iteritems():
            try:
                print "\t{}: {:.2f} +/- {:.2f}".format(name, dist.mean(), dist.std())
            except TypeError:
                print "\t{}: {}".format(name, collections.Counter(dist).most_common(1))

    def get_feature_importances(self, feature_names):
        feature_importances = collections.defaultdict(list)

        for estimator in self.estimators_:
            try:
                importances = estimator.feature_importances_
            except AttributeError:
                try:
                    importances = estimator.best_estimator_.feature_importances_
                except AttributeError:
                    raise ValueError("Unable to find feature_importances_")

            for feature_name, feature_score in zip(feature_names, importances):
                feature_importances[feature_name].append(feature_score)

        return {k: numpy.asarray(v) for k, v in feature_importances.iteritems()}

    def print_feature_importances(self, feature_names):
        print "Feature importances"

        scores = self.get_feature_importances(feature_names)
        for name, dist in sorted(scores.iteritems(), key=lambda pair: pair[1].mean(), reverse=True):
            print "\t{}: {:.3f} +/- {:.3f}".format(name, dist.mean(), dist.std())


def print_tuning_scores(tuned_estimator, reverse=True, score_transformer=None):
    """Show the cross-validation scores and hyperparamters from a grid or random search"""
    for test in sorted(tuned_estimator.grid_scores_, key=itemgetter(1), reverse=reverse):
        scores = test.cv_validation_scores
        if score_transformer:
            scores = score_transformer(scores)
        print "Validation score {:.4f} +/- {:.4f}, Hyperparams {}".format(scores.mean(),
                                                                          scores.std(),
                                                                          test.parameters)


def mse_to_rms(scores):
    return numpy.sqrt(numpy.abs(scores))


def print_feature_importances(columns, classifier):
    """Show feature importances for a classifier that supports them like random forest or gradient boosting"""
    paired_features = zip(columns, classifier.feature_importances_)
    field_width = unicode(max(len(c) for c in columns))
    format_string = "\t{:" + field_width + "s}: {}"
    print "Feature importances"
    for feature_name, importance in sorted(paired_features, key=itemgetter(1), reverse=True):
        print format_string.format(feature_name, importance)


class RandomizedSearchCV(sklearn.grid_search.RandomizedSearchCV):
    def __init__(self, *args, **kwargs):
        super(RandomizedSearchCV, self).__init__(*args, **kwargs)

    def print_tuning_scores(self, score_transformer=None, reverse=True):
        for test in sorted(self.grid_scores_, key=itemgetter(1), reverse=reverse):
            scores = test.cv_validation_scores
            if score_transformer:
                scores = score_transformer(scores)
            print "Validation score {:.4f} +/- {:.4f}, Hyperparams {}".format(scores.mean(),
                                                                              scores.std(),
                                                                              test.parameters)

        print "Hyperparameter correlations with evaluation metric"
        for param, correlation_r in self.correlate_hyperparameters(score_transformer=score_transformer):
            print "{}: {:.4f}".format(param, correlation_r)

    def correlate_hyperparameters(self, score_transformer=None):
        param_scores = collections.defaultdict(list)
        for test in self.grid_scores_:
            scores = test.cv_validation_scores
            if score_transformer:
                scores = score_transformer(scores)

            for name, value in test.parameters.iteritems():
                if isinstance(value, numbers.Number):
                    param_scores[name].append((value, scores.mean()))

        param_correlations = collections.Counter()
        for param_name, points in param_scores.iteritems():
            points = numpy.asarray(points)
            assert points.shape[1] == 2

            pearson_r, pearson_p = scipy.stats.pearsonr(points[:, 0], points[:, 1])
            param_correlations[param_name] = pearson_r

        return param_correlations


class LinearRegressionWrapper(sklearn.linear_model.LinearRegression):
    """Wrapper for LinearRegression that's compatible with GradientBoostingClassifier sample_weights"""
    def fit(self, X, y, sample_weight, **kwargs):
        super(LinearRegressionWrapper, self).fit(X, y, **kwargs)

    def predict(self, X):
        return super(LinearRegressionWrapper, self).predict(X)[:, numpy.newaxis]

class NnRegressor(sklearn.base.BaseEstimator):
    """Wrapper for Keras feed-forward neural network for classification to enable scikit-learn grid search"""
    def __init__(self, hidden_layer_sizes=(100,), dropout=0.5, batch_spec=((400, 1024), (100, -1)), hidden_activation="relu", input_noise=0., use_maxout=False, use_maxnorm=False, learning_rate=0.001, verbose=0, init="he_uniform"):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.batch_spec = batch_spec
        self.hidden_activation = hidden_activation
        self.input_noise = input_noise
        self.use_maxout = use_maxout
        self.use_maxnorm = use_maxnorm
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.loss = "mse"
        self.init = init

        if self.use_maxout:
            self.use_maxnorm = True

        self.model_ = None

    def fit(self, X, y, **kwargs):
        self.set_params(**kwargs)

        if self.verbose >= 1:
            print "Fitting input shape {}, output shape {}".format(X.shape, y.shape)

        model = keras.models.Sequential()

        first = True

        if self.input_noise > 0:
            model.add(keras.layers.noise.GaussianNoise(self.input_noise, input_shape=X.shape[1:]))

        num_maxout_features = 2

        dense_kwargs = {"init": self.init}
        if self.use_maxnorm:
            dense_kwargs["W_constraint"] = keras.constraints.maxnorm(2)

        # hidden layers
        for layer_size in self.hidden_layer_sizes:
            if first:
                if self.use_maxout:
                    model.add(keras.layers.core.MaxoutDense(output_dim=layer_size / num_maxout_features, input_dim=X.shape[1], init=self.init, nb_feature=num_maxout_features))
                else:
                    model.add(keras.layers.core.Dense(output_dim=layer_size, input_dim=X.shape[1], **dense_kwargs))
                    model.add(keras.layers.core.Activation(self.hidden_activation))
                first = False
            else:
                if self.use_maxout:
                    model.add(keras.layers.core.MaxoutDense(output_dim=layer_size / num_maxout_features, init=self.init, nb_feature=num_maxout_features))
                else:
                    model.add(keras.layers.core.Dense(output_dim=layer_size, **dense_kwargs))
                    model.add(keras.layers.core.Activation(self.hidden_activation))
            model.add(keras.layers.core.Dropout(self.dropout))

        if first:
            model.add(keras.layers.core.Dense(output_dim=y.shape[1], input_dim=X.shape[1], **dense_kwargs))
        else:
            model.add(keras.layers.core.Dense(output_dim=y.shape[1], **dense_kwargs))

        optimizer = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(loss=self.loss, optimizer=optimizer)

        # batches as per configuration
        for num_iterations, batch_size in self.batch_spec:
            fit_kwargs = {"verbose": self.verbose}

            if batch_size < 0:
                batch_size = X.shape[0]
            elif batch_size > X.shape[0]:
                print "Clipping batch size to input rows"
                batch_size = X.shape[0]

            if num_iterations > 0:
                model.fit(X, y, nb_epoch=num_iterations, batch_size=batch_size, **fit_kwargs)

        self.model_ = model
        return self

    def count_params(self):
        return self.model_.count_params()

    def predict(self, X):
        # sklearn VotingClassifier requires this to be 1-dimensional
        return self.model_.predict(X)

    # def score(self, X, y):
    #     return sklearn.metrics.accuracy_score(y, self.predict(X))

    @staticmethod
    def generate_batch_params(mini_batch_iter, total_epochs=200, mini_batch_size=1024):
        for mini_batch_epochs in mini_batch_iter:
            assert mini_batch_epochs <= total_epochs
            yield ((mini_batch_epochs, mini_batch_size), (total_epochs - mini_batch_epochs, -1))

    @staticmethod
    def compute_num_params(X, layer_spec):
        layers = [X.shape[1]] + list(layer_spec) + [1]

        num_params = 0
        for i in xrange(1, len(layers)):
            num_params += (layers[i - 1] + 1) * layers[i]

        return num_params
