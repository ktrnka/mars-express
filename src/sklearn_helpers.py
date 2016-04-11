from __future__ import unicode_literals

import math
import numbers
import sys
import argparse
from operator import itemgetter

import collections

import time

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
import keras.layers.advanced_activations

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

        print "Linear hyperparameter correlations with evaluation metric"
        for param, (stat_name, stat, pval) in self.correlate_hyperparameters(score_transformer=score_transformer).iteritems():
            print "\t{}: {} = {:.4f}, p = {:.4f}".format(param, stat_name, stat, pval)

        # print "Folded hyperparameter correlations with evaluation metric"
        # for param, (stat_name, stat, pval) in self.correlate_hyperparameters(score_transformer=score_transformer, fold_over_max=True).iteritems():
        #     print "\t{}: {} = {:.4f}, p = {:.4f}".format(param, stat_name, stat, pval)

    def correlate_hyperparameters(self, score_transformer=None, fold_over_max=False):
        param_scores = self._get_independent_scores(score_transformer)

        param_correlations = dict()
        for param_name, points in param_scores.iteritems():
            if all(isinstance(x, numbers.Number) for x, _ in points):
                # numeric params path: use Pearson
                points = numpy.asarray(points)
                assert points.shape[1] == 2

                if fold_over_max:
                    _, max_score_index = numpy.argmax(points, axis=0)
                    points[:, 0] = numpy.abs(points[:, 0] - points[max_score_index, 0])

                pearson_r, pearson_p = scipy.stats.pearsonr(points[:, 0], points[:, 1])
                param_correlations[param_name] = ("Pearson r", pearson_r, pearson_p)
            else:
                # non-numeric path, run anova or something
                param_vals = collections.defaultdict(list)
                for param_val, score in points:
                    param_vals[param_val].append(score)

                anova_f, anova_p = scipy.stats.f_oneway(*[numpy.asarray(v) for v in param_vals.itervalues()])
                param_correlations[param_name] = ("Anova f", anova_f, anova_p)

        return param_correlations

    def _get_independent_scores(self, score_transformer):
        param_scores = collections.defaultdict(list)
        param_counts = collections.defaultdict(collections.Counter)
        for test in self.grid_scores_:
            scores = test.cv_validation_scores
            if score_transformer:
                scores = score_transformer(scores)

            for name, value in test.parameters.iteritems():
                param_counts[name][value] += 1
                param_scores[name].append((value, scores.mean()))

        # remove parameter values that don't vary
        for param, value_distribution in param_counts.iteritems():
            if len(value_distribution) == 1 and param in param_scores:
                del param_scores[param]

        return param_scores

    @staticmethod
    def uniform(start, end):
        """Helper to make a continuous or discrete uniform distribution depending on the input types"""
        if all(isinstance(x, int) for x in [start, end]):
            return scipy.stats.randint(start, end)
        else:
            return scipy.stats.uniform(start, end - start)

    @staticmethod
    def exponential(start, end, num_samples=100):
        """Helper to make a log-linear distribution"""
        return numpy.exp(numpy.linspace(math.log(start), math.log(end), num=num_samples))



class LinearRegressionWrapper(sklearn.linear_model.LinearRegression):
    """Wrapper for LinearRegression that's compatible with GradientBoostingClassifier sample_weights"""
    def fit(self, X, y, sample_weight, **kwargs):
        super(LinearRegressionWrapper, self).fit(X, y, **kwargs)

    def predict(self, X):
        return super(LinearRegressionWrapper, self).predict(X)[:, numpy.newaxis]

class NnRegressor(sklearn.base.BaseEstimator):
    """Wrapper for Keras feed-forward neural network for classification to enable scikit-learn grid search"""
    def __init__(self, hidden_layer_sizes=(100,), dropout=0.5, batch_spec=((400, 1024), (100, -1)), hidden_activation="relu", input_noise=0., use_maxout=False, use_maxnorm=False, learning_rate=0.001, verbose=0, init="he_uniform", l2=None):
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
        self.l2 = l2

        if self.use_maxout:
            self.use_maxnorm = True

        self.model_ = None

    def _get_activation(self):
        if self.hidden_activation == "elu":
            return keras.layers.advanced_activations.ELU()
        else:
            return keras.layers.core.Activation(self.hidden_activation)

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
        if self.l2:
            dense_kwargs["W_regularizer"] = keras.regularizers.l2(self.l2)


        # hidden layers
        for layer_size in self.hidden_layer_sizes:
            if first:
                if self.use_maxout:
                    model.add(keras.layers.core.MaxoutDense(output_dim=layer_size / num_maxout_features, input_dim=X.shape[1], init=self.init, nb_feature=num_maxout_features))
                else:
                    model.add(keras.layers.core.Dense(output_dim=layer_size, input_dim=X.shape[1], **dense_kwargs))
                    model.add(self._get_activation())
                first = False
            else:
                if self.use_maxout:
                    model.add(keras.layers.core.MaxoutDense(output_dim=layer_size / num_maxout_features, init=self.init, nb_feature=num_maxout_features))
                else:
                    model.add(keras.layers.core.Dense(output_dim=layer_size, **dense_kwargs))
                    model.add(self._get_activation())
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
        return fill_nan(self.model_.predict(X))

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


class ExtraRobustScaler(sklearn.preprocessing.RobustScaler):
    def transform(self, X, y=None):
        X = super(ExtraRobustScaler, self).transform(X, y)

        # try to ensure that -1 to 1 is a nice linear range and squash a bit beyond that
        X[X > 5] = 5
        X[X < -5] = -5

        return X

def number_string(number, singular_unit, plural_unit, format_string="{} {}"):
    return format_string.format(number, singular_unit if number == 1 else plural_unit)


class Timed(object):
    """Decorator for timing how long a function takes"""
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        self.func(*args, **kwargs)
        elapsed = time.time() - start_time

        hours, seconds = divmod(elapsed, 60 * 60)
        minutes = seconds / 60.
        time_string = number_string(minutes, "minute", "minutes", format_string="{:.1f} {}")
        if hours:
            time_string = ", ".join((number_string(hours, "hour", "hours"), time_string))

        print "{} took {}".format(self.func.__name__, time_string)


class TimeCV(object):
    """
    Cross-validation wrapper for time-series prediction, i.e., test only on extrapolations into the future.
    Assumes that the data is sorted chronologically.
    """
    def __init__(self, num_rows, num_splits, min_training=0.5, test_min_splits=1, cheap_reverse=False, gap=0, balanced_tests=True):
        self.num_rows = int(num_rows)
        self.num_splits = int(num_splits)
        self.test_split_buckets = test_min_splits
        self.min_training = min_training

        self.cheap_reverse = cheap_reverse
        self.gap = gap
        self.balanced_tests = balanced_tests

    def __iter__(self):
        per_bin = self.num_rows / float(self.num_splits)

        for s in xrange(1, self.num_splits):
            train_end = int(per_bin * s)

            test_start = train_end + int(self.gap * per_bin)
            test_end = test_start + int(per_bin * self.test_split_buckets)

            if not self.balanced_tests:
                test_end = min(test_end, self.num_rows)

                # sometimes one leftover due to rounding error
                if test_end - test_start <= 1:
                    continue

            # only return uniform size tests
            if self.balanced_tests and test_end > self.num_rows:
                continue

            train_index = numpy.asarray(range(0, train_end), dtype=numpy.int32)
            test_index = numpy.asarray(range(test_start, test_end), dtype=numpy.int32)

            # skip any without enough data
            if train_end >= int(self.min_training * self.num_rows):
                yield list(train_index), list(test_index)

                if self.cheap_reverse:
                    yield list(self.num_rows - train_index - 1), list(self.num_rows - test_index - 1)


def fill_nan(a, method="mean"):
    df = pandas.DataFrame(a)

    nan_count = df.isnull().sum().sum()
    print "Replacing {:,} / {:,} null values".format(nan_count, df.shape[0] * df.shape[1])
    df = df.fillna(df.mean())
    return df.values
