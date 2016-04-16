from __future__ import unicode_literals

import logging

import keras.callbacks
import keras.constraints
import keras.layers
import keras.models
import keras.optimizers
import keras.regularizers
import numpy
import pandas
import sklearn
import keras.layers.recurrent
import helpers.general
import sklearn.utils
import pandas

_he_activations = {"relu"}


class NnRegressor(sklearn.base.BaseEstimator):
    """Wrapper for Keras feed-forward neural network for regression to enable scikit-learn grid search"""

    def __init__(self, hidden_layer_sizes=(100,), hidden_units=None, dropout=None, batch_size=-1, loss="mse", num_epochs=500, activation="relu", input_noise=0., learning_rate=0.001, verbose=0, init=None, l2=None, batch_norm=False, early_stopping=False, clip_gradient_norm=None, assert_finite=True,
                 maxnorm=False, val=0., history_file=None):
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

        self.logger = logging.getLogger(type(self).__name__)

        self.model_ = None

    def _get_activation(self):
        if self.activation == "elu":
            return keras.layers.advanced_activations.ELU()
        else:
            return keras.layers.core.Activation(self.activation)

    def fit(self, X, y, **kwargs):
        self.set_params(**kwargs)

        if self.verbose >= 1:
            print "Fitting input shape {}, output shape {}".format(X.shape, y.shape)

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

        optimizer = keras.optimizers.Adam(**self._get_optimizer_kwargs())
        model.compile(loss=self.loss, optimizer=optimizer)

        self._save_history(model.fit(X, y, **self._get_fit_kwargs(X)))

        self.model_ = model
        self.logger.info("Model has {:,} params".format(self.count_params()))
        return self

    def _get_dense_layer_kwargs(self):
        """Apply settings to dense layer keyword args"""
        dense_kwargs = {"init": self.init}
        if self.l2:
            dense_kwargs["W_regularizer"] = keras.regularizers.l2(self.l2)

        if self.use_maxnorm:
            dense_kwargs["W_constraint"] = keras.constraints.MaxNorm(2)
            dense_kwargs["b_constraint"] = keras.constraints.MaxNorm(2)

        return dense_kwargs

    def _get_fit_kwargs(self, X):
        """Apply settings to the fit function keyword args"""
        kwargs = {"verbose": self.verbose, "nb_epoch": self.num_epochs, "callbacks": []}

        if self.early_stopping:
            monitor = "val_loss" if self.val > 0 else "loss"
            es = keras.callbacks.EarlyStopping(monitor=monitor, patience=self.num_epochs / 20, verbose=self.verbose, mode="min")
            kwargs["callbacks"].append(es)

        if self.val > 0:
            kwargs["validation_split"] = self.val

        kwargs["batch_size"] = self.batch_size
        if kwargs["batch_size"] < 0 or kwargs["batch_size"] > X.shape[0]:
            kwargs["batch_size"] = X.shape[0]

        return kwargs

    def count_params(self):
        return self.model_.count_params()

    def predict(self, X):
        return self._check_finite(self.model_.predict(X))

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
                 early_stopping=False, dropout=None, recurrent_dropout=None, loss="mse", input_noise=0., learning_rate=0.001, clip_gradient_norm=None, val=0, assert_finite=True, history_file=None):
        super(RnnRegressor, self).__init__(batch_size=batch_size, num_epochs=num_epochs, verbose=verbose, early_stopping=early_stopping, dropout=dropout, loss=loss, input_noise=input_noise, learning_rate=learning_rate, clip_gradient_norm=clip_gradient_norm, val=val, assert_finite=assert_finite, history_file=history_file)
        self.num_units = num_units
        self.time_steps = time_steps
        self.unit = unit
        self.recurrent_dropout = recurrent_dropout
        self.use_maxnorm = True

        self.logger = logging.getLogger(type(self).__name__)

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

        self.logger.info("X takes %d mb", X.nbytes / 10e6)
        self.logger.info("X_time takes %d mb", X_time.nbytes / 10e6)

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

        optimizer = keras.optimizers.RMSprop(**self._get_optimizer_kwargs())
        model.compile(loss="mse", optimizer=optimizer)

        self._save_history(model.fit(X_time, Y, **self._get_fit_kwargs(X)))

        self.model_ = model

        self.logger.info("Model has {:,} params".format(self.count_params()))

        return self

    def predict(self, X):
        return self._check_finite(self.model_.predict(self._transform_input(X)))


def make_learning_rate_schedule(initial_value, exponential_decay=1.):
    def schedule(epoch_num):
        return initial_value * exponential_decay ** epoch_num

    return schedule