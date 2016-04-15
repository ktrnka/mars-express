from __future__ import unicode_literals

import keras.callbacks
import keras.constraints
import keras.layers
import keras.models
import keras.optimizers
import keras.regularizers
import numpy
import sklearn
import keras.layers.recurrent
import helpers.general

_he_activations = {"relu"}


class RnnSpec(object):
    def __init__(self, num_units=None):
        self.num_units = num_units

    def get_unit_first(self, batch_size, num_features):
        return keras.layers.recurrent.LSTM(self.num_units, stateful=True, batch_input_shape=(batch_size, 1, num_features))

class NnRegressor(sklearn.base.BaseEstimator):
    """Wrapper for Keras feed-forward neural network for regression to enable scikit-learn grid search"""

    def __init__(self, hidden_layer_sizes=(100,), hidden_units=None, dropout=0.5, batch_size=-1, loss="mse", num_epochs=500, activation="relu", input_noise=0., learning_rate=0.001, verbose=0, init=None, l2=None, batch_norm=False, early_stopping=False, clip_gradient_norm=None, assert_finite=True,
                 maxnorm=False, rnn_spec=None):
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
        self.rnn_spec = rnn_spec

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

        # TODO: I made an incorrect conclusion about input_shape because I always had input_noise enabled before!

        if self.rnn_spec:
            model.add(self.rnn_spec.get_unit_first(self.batch_size, X.shape[-1]))

        # optional input noise
        if self.input_noise > 0:
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

        model.fit(X, y, nb_epoch=self.num_epochs, **self._get_fit_kwargs(X))

        self.model_ = model
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
        kwargs = {"verbose": self.verbose, "callbacks": []}

        if self.early_stopping:
            es = keras.callbacks.EarlyStopping(monitor="loss", patience=self.num_epochs / 20, verbose=self.verbose, mode="min")
            kwargs["callbacks"].append(es)

        kwargs["batch_size"] = self.batch_size
        if kwargs["batch_size"] < 0 or kwargs["batch_size"] > X.shape[0]:
            kwargs["batch_size"] = X.shape[0]

        if self.rnn_spec:
            kwargs["shuffle"] = False

        return kwargs

    def count_params(self):
        return self.model_.count_params()

    def predict(self, X):
        Y = self.model_.predict(X)

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


class RnnRegressor(sklearn.base.BaseEstimator):
    def __init__(self, num_units=50, time_steps=5, batch_size=100, num_epochs=100):
        self.num_units = num_units
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def _transform_input(self, X):
        return helpers.general.prepare_time_matrix(X, self.time_steps, fill_value=0)

    def fit(self, X, Y):
        model = keras.models.Sequential()

        X_time = self._transform_input(X)

        # hidden layer
        model.add(keras.layers.recurrent.LSTM(self.num_units, batch_input_shape=(self.batch_size, self.time_steps, X.shape[1])))

        # output layer
        model.add(keras.layers.core.Dense(output_dim=Y.shape[1]))

        model.compile(loss="mse", optimizer="rmsprop")

        model.fit(X_time, Y, nb_epoch=self.num_epochs, verbose=2)

        self.model_ = model
        return self

    def predict(self, X):
        return self.model_.predict(self._transform_input(X))

