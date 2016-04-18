from __future__ import unicode_literals

import time

import keras.callbacks
import keras.constraints
import keras.layers
import keras.layers.recurrent
import keras.models
import keras.optimizers
import keras.regularizers
import numpy
import pandas
import sklearn
import sklearn.utils
import theano
import os

import helpers.general

_he_activations = {"relu"}


def set_theano_float_precision(precision):
    assert precision in {"float32", "float64"}
    theano.config.floatX = precision


def disable_theano_gc():
    theano.config.allow_gc = False


def enable_openmp():
    print("Current OpenMP value:", theano.config.openmp)
    theano.config.openmp = True
    os.environ["OMP_NUM_THREADS"] = "2"


class NnRegressor(sklearn.base.BaseEstimator):
    """Wrapper for Keras feed-forward neural network for regression to enable scikit-learn grid search"""

    def __init__(self, hidden_layer_sizes=(100,), hidden_units=None, dropout=None, batch_size=-1, loss="mse", num_epochs=500, activation="relu", input_noise=0., learning_rate=0.001, verbose=0, init=None, l2=None, batch_norm=False, early_stopping=False, clip_gradient_norm=None, assert_finite=True,
                 maxnorm=False, val=0., history_file=None, optimizer="adam", schedule=None):
        self.clip_gradient_norm = clip_gradient_norm
        self.assert_finite = assert_finite
        if hidden_units:
            self.hidden_layer_sizes = (hidden_units,)
        else:
            self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.activation = activation
        self.input_noise = input_noise
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.loss = loss
        self.l2 = l2
        self.batch_norm = batch_norm
        self.early_stopping = early_stopping
        self.init = self._get_default_init(init, activation)
        self.use_maxnorm = maxnorm
        self.val = val
        self.history_file = history_file
        self.optimizer = optimizer
        self.schedule = schedule
        self.extra_callback = None

        self.logger = helpers.general.get_class_logger(self)

        self.model_ = None

    def _get_optimizer(self):
        if self.optimizer == "adam":
            return keras.optimizers.Adam(**self._get_optimizer_kwargs())
        elif self.optimizer == "rmsprop":
            return keras.optimizers.RMSprop(**self._get_optimizer_kwargs())
        elif self.optimizer == "sgd":
            return keras.optimizers.SGD(**self._get_optimizer_kwargs())
        elif self.optimizer == "adamax":
            return keras.optimizers.Adamax(**self._get_optimizer_kwargs())
        else:
            raise ValueError("Unknown optimizer {}".format(self.optimizer))

    def _get_activation(self):
        if self.activation == "elu":
            return keras.layers.advanced_activations.ELU()
        else:
            return keras.layers.core.Activation(self.activation)

    def fit(self, X, y, **kwargs):
        self.set_params(**kwargs)

        self.logger.debug("X: {}, Y: {}".format(X.shape, y.shape))

        model = keras.models.Sequential()

        # input noise not optional so that we have a well-defined first layer to
        # set the input shape on (though it may be set to zero noise)
        model.add(keras.layers.noise.GaussianNoise(self.input_noise, input_shape=X.shape[1:]))

        dense_kwargs = self._get_dense_layer_kwargs()

        # hidden layers
        for layer_size in self.hidden_layer_sizes:
            model.add(keras.layers.core.Dense(output_dim=layer_size, **dense_kwargs))
            if self.batch_norm:
                model.add(keras.layers.normalization.BatchNormalization())
            model.add(self._get_activation())

            if self.dropout:
                model.add(keras.layers.core.Dropout(self.dropout))

        # output layer
        model.add(keras.layers.core.Dense(output_dim=y.shape[1], **dense_kwargs))

        optimizer = self._get_optimizer()
        model.compile(loss=self.loss, optimizer=optimizer)

        self.model_ = model
        self._run_fit(X, y)

        return self

    def _run_fit(self, X, y):
        t = time.time()
        history = self.model_.fit(X, y, **self._get_fit_kwargs(X))
        t = time.time() - t

        self._save_history(history)

        self.logger.info("Trained at {:,} rows/sec in {:,} epochs".format(int(X.shape[0] * len(history.epoch) / t), len(history.epoch)))
        self.logger.debug("Model has {:,} params".format(self.count_params()))

    def _get_dense_layer_kwargs(self):
        """Apply settings to dense layer keyword args"""
        dense_kwargs = {"init": self.init}
        if self.l2:
            dense_kwargs["W_regularizer"] = keras.regularizers.l2(self.l2)

        if self.use_maxnorm:
            dense_kwargs["W_constraint"] = keras.constraints.MaxNorm(2)
            dense_kwargs["b_constraint"] = keras.constraints.MaxNorm(2)

        return dense_kwargs

    def _get_fit_kwargs(self, X, batch_size_override=None, num_epochs_override=None):
        """Apply settings to the fit function keyword args"""
        kwargs = {"verbose": self.verbose, "nb_epoch": self.num_epochs, "callbacks": []}

        if num_epochs_override:
            kwargs["nb_epoch"] = num_epochs_override

        if self.early_stopping:
            monitor = "val_loss" if self.val > 0 else "loss"
            es = keras.callbacks.EarlyStopping(monitor=monitor, patience=self.num_epochs / 20, verbose=self.verbose, mode="min")
            kwargs["callbacks"].append(es)

        if self.schedule:
            kwargs["callbacks"].append(keras.callbacks.LearningRateScheduler(self.schedule))

        if self.extra_callback:
            kwargs["callbacks"].append(self.extra_callback)

        if self.val > 0:
            kwargs["validation_split"] = self.val

        kwargs["batch_size"] = self.batch_size
        if batch_size_override:
            kwargs["batch_size"] = batch_size_override
        if kwargs["batch_size"] < 0 or kwargs["batch_size"] > X.shape[0]:
            kwargs["batch_size"] = X.shape[0]

        self.logger.info("Fit kwargs: %s", kwargs)

        return kwargs

    def count_params(self):
        return self.model_.count_params()

    def predict(self, X):
        retval = self._check_finite(self.model_.predict(X))
        return retval

    def _check_finite(self, Y):
        if self.assert_finite:
            sklearn.utils.assert_all_finite(Y)
        else:
            Y = numpy.nan_to_num(Y)

        return Y

    def _get_default_init(self, init, activation):
        if init:
            return init

        if activation in _he_activations:
            return "he_uniform"

        return "glorot_uniform"

    def _get_optimizer_kwargs(self):
        kwargs = {"lr": self.learning_rate}

        if self.clip_gradient_norm:
            kwargs["clipnorm"] = self.clip_gradient_norm

        return kwargs

    def _save_history(self, history):
        if not self.history_file:
            return

        dataframe = pandas.DataFrame.from_dict(history.history)
        dataframe.index.rename("epoch", inplace=True)
        dataframe.to_csv(self.history_file)


class RnnRegressor(NnRegressor):
    def __init__(self, num_units=50, time_steps=5, batch_size=100, num_epochs=100, unit="lstm", verbose=0,
                 early_stopping=False, dropout=None, recurrent_dropout=None, loss="mse", input_noise=0., learning_rate=0.001, clip_gradient_norm=None, val=0, assert_finite=True, history_file=None,
                 pretrain=True, optimizer="adam"):
        super(RnnRegressor, self).__init__(batch_size=batch_size, num_epochs=num_epochs, verbose=verbose, early_stopping=early_stopping, dropout=dropout, loss=loss, input_noise=input_noise, learning_rate=learning_rate, clip_gradient_norm=clip_gradient_norm, val=val, assert_finite=assert_finite, history_file=history_file, optimizer=optimizer)
        self.num_units = num_units
        self.time_steps = time_steps
        self.unit = unit
        self.recurrent_dropout = recurrent_dropout
        self.use_maxnorm = True
        self.pretrain = pretrain

        self.logger = helpers.general.get_class_logger(self)

    def _transform_input(self, X):
        return helpers.general.prepare_time_matrix(X, self.time_steps, fill_value=0)

    def _get_recurrent_layer_kwargs(self):
        """Apply settings to dense layer keyword args"""
        kwargs = {"output_dim": self.num_units}

        if self.recurrent_dropout:
            kwargs["dropout_U"] = self.recurrent_dropout

        return kwargs

    def fit(self, X, Y, **kwargs):
        self.set_params(**kwargs)

        model = keras.models.Sequential()

        X_time = self._transform_input(X)

        self.logger.debug("X takes %d mb", X.nbytes / 10e6)
        self.logger.debug("X_time takes %d mb", X_time.nbytes / 10e6)

        model.add(keras.layers.noise.GaussianNoise(self.input_noise, input_shape=X_time.shape[1:]))

        # hidden layer
        if self.unit == "lstm":
            model.add(keras.layers.recurrent.LSTM(**self._get_recurrent_layer_kwargs()))
        elif self.unit == "gru":
            model.add(keras.layers.recurrent.GRU(**self._get_recurrent_layer_kwargs()))
        else:
            raise ValueError("Unknown unit type: {}".format(self.unit))

        # dropout
        if self.dropout:
            model.add(keras.layers.core.Dropout(self.dropout))

        # output layer
        model.add(keras.layers.core.Dense(output_dim=Y.shape[1], **self._get_dense_layer_kwargs()))

        optimizer = self._get_optimizer()
        model.compile(loss="mse", optimizer=optimizer)
        self.model_ = model

        if self.pretrain:
            self.model_.fit(X_time, Y, **self._get_fit_kwargs(X, batch_size_override=1, num_epochs_override=1))

        self._run_fit(X_time, Y)

        return self

    def predict(self, X):
        r = self._check_finite(self.model_.predict(self._transform_input(X)))
        return r


def make_learning_rate_schedule(initial_value, exponential_decay=1., kick_every=10000):
    logger = helpers.general.get_function_logger()

    def schedule(epoch_num):
        lr = initial_value * (10 ** int(epoch_num / kick_every)) * exponential_decay ** epoch_num
        logger.info("Setting learning rate at {} to {}".format(epoch_num, lr))
        return lr

    return schedule

import keras.backend


class VarianceLearningRateScheduler(keras.callbacks.Callback):
    def __init__(self, initial_lr=0.01, monitor="val_loss", scale=2):
        super(VarianceLearningRateScheduler, self).__init__()
        self.monitor = monitor
        self.initial_lr = initial_lr
        self.scale = float(scale)

        self.metric_ = []
        self.logger_ = helpers.general.get_class_logger(self)

    def on_epoch_begin(self, epoch, logs={}):
        assert hasattr(self.model.optimizer, 'lr'), 'Optimizer must have a "lr" attribute.'

        lr = self._get_learning_rate()

        if lr:
            self.logger_.info("Setting learning rate at %d to %e", epoch, lr)
            keras.backend.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs={}):
        metric = logs[self.monitor]
        self.metric_.append(metric)

    def _get_learning_rate(self):
        window = 3
        if len(self.metric_) < window * 2:
            return self.initial_lr

        data = numpy.asarray(self.metric_)

        baseline = data[:-window].min()
        diffs = baseline - data[-window:]

        # assume error, lower is better
        percent_epochs_improved = (diffs > 0).mean()
        self.logger_.info("Ratio of good epochs: %.2f", percent_epochs_improved)

        if percent_epochs_improved > 0.75:
            return self._scale_learning_rate(self.scale)
        elif percent_epochs_improved < 0.5:
            return self._scale_learning_rate(1. / self.scale)

        return None

    def _scale_learning_rate(self, scale):
        return keras.backend.get_value(self.model.optimizer.lr) * scale
