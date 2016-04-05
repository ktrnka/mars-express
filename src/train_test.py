from __future__ import unicode_literals
import sys
import argparse

import collections

import pandas
import sklearn
import numpy
import sklearn.cross_validation
import sklearn.preprocessing
import sklearn.dummy
import sklearn.linear_model
import sklearn.ensemble
import sklearn.grid_search
import os
import scipy.stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-analysis", default=False, action="store_true", help="Extra analysis on the data")
    parser.add_argument("training_dir", help="Dir with the training CSV files")
    parser.add_argument("testing_dir", help="Dir with the testing files, including the empty prediction file")
    parser.add_argument("prediction_file", help="Destination for predictions")
    return parser.parse_args()


def parse_dates(column):
    return pandas.to_datetime(column.astype(int), unit="ms")


def find_files(path, substring):
    files = os.listdir(path)
    return sorted(os.path.join(path, f) for f in files if substring in f)


def load_series(files, add_file_number=False, resample_interval=None):
    data = [pandas.read_csv(f, parse_dates=["ut_ms"], date_parser=parse_dates, index_col=0) for f in files]

    if resample_interval:
        data = [d.resample(resample_interval).mean() for d in data]

    if add_file_number:
        for i, year_data in enumerate(data):
            year_data["file_number"] = i

    return pandas.concat(data)


def load_data(data_dir, resample_interval=None):
    # load the base power data
    data = load_series(find_files(data_dir, "power"), add_file_number=True, resample_interval=resample_interval)

    saaf_data = load_series(find_files(data_dir, "saaf"))
    longterm_data = load_series(find_files(data_dir, "ltdata"))

    # longterm_lagged = longterm_data.rolling(7).mean().fillna(method="bfill")
    # longterm_data = longterm_data.merge(longterm_lagged, left_index=True, right_index=True, suffixes=("", "_rolling7"))

    for other_data in [saaf_data, longterm_data]:
        data = pandas.concat([data, other_data.reindex(data.index, method="nearest")], axis=1)

    data["days_in_space"] = (data.index - pandas.datetime(year=2003, month=6, day=2)).days

    return data.fillna(data.mean())


def mse_to_rms(scores):
    return numpy.sqrt(-scores)


def compute_upper_bounds(data):
    data = data[[c for c in data.columns if c.startswith("NPWD")]]

    for interval in "7D 1D 12H 6H 2H 1H 30M".split():
        downsampled_data = data.resample(interval).mean()
        upsampled_data = downsampled_data.reindex(data.index, method="pad")

        rms = ((data - upsampled_data) ** 2).mean().mean() ** 0.5
        print "RMS with {} approximation: {:.3f}".format(interval, rms)

class VectorRegression(sklearn.base.BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator
        self.estimators_ = None

    def fit(self, X, y):
        n, m = y.shape
        # Fit a separate regressor for each column of y
        self.estimators_ = [sklearn.base.clone(self.estimator).fit(X, y[:, i]) for i in range(m)]
        return self

    def predict(self, X):
        # Join regressors' predictions
        res = [est.predict(X)[:, numpy.newaxis] for est in self.estimators_]
        return numpy.hstack(res)

def main():
    args = parse_args()

    train_data = load_data(args.training_dir, resample_interval="1H")

    # cross validation by year
    splits = sklearn.cross_validation.LeaveOneLabelOut(train_data["file_number"])

    # just use the biggest one for now
    X_train, Y_train = separate_output(train_data)

    if args.extra_analysis:
        X_train.info()
        print X_train.describe()

    scaler = sklearn.preprocessing.RobustScaler()
    X_train = scaler.fit_transform(X_train)

    # lower bound: predict mean
    baseline_model = sklearn.dummy.DummyRegressor("mean")
    baseline_scores = mse_to_rms(sklearn.cross_validation.cross_val_score(baseline_model, X_train, Y_train, scoring="mean_squared_error", cv=splits))
    print "DummyRegressor(mean): {:.3f} +/- {:.3f}".format(baseline_scores.mean(), baseline_scores.std())

    # upper bound: predict average per diem
    if args.extra_analysis:
        compute_upper_bounds(train_data)

    model = sklearn.linear_model.LinearRegression()
    scores = mse_to_rms(sklearn.cross_validation.cross_val_score(model, X_train, Y_train, scoring="mean_squared_error", cv=splits))
    print "LinearRegression: {:.3f} +/- {:.3f}".format(scores.mean(), scores.std())

    # model = sklearn.linear_model.PassiveAggressiveRegressor()
    # scores = sklearn.cross_validation.cross_val_score(model, X, Y, scoring="mean_squared_error", cv=splits)
    # print "PA: {:.3f} +/- {:.3f}".format(scores.mean(), scores.std())

    model = sklearn.ensemble.RandomForestRegressor(20, min_samples_leaf=100, max_depth=3)
    scores = mse_to_rms(sklearn.cross_validation.cross_val_score(model, X_train, Y_train, scoring="mean_squared_error", cv=splits))
    print "RandomForestRegression: {:.3f} +/- {:.3f}".format(scores.mean(), scores.std())

    # rf_hyperparams = {
    #     "min_samples_leaf": scipy.stats.randint(10, 100),
    #     "max_depth": scipy.stats.randint(3, 10),
    #     "max_features": ["sqrt", "log2", 1.0, 0.3],
    #     "n_estimators": scipy.stats.randint(20, 100)
    # }
    # wrapped_model = sklearn.grid_search.RandomizedSearchCV(model, rf_hyperparams, n_iter=10, n_jobs=3)
    # scores = mse_to_rms(sklearn.cross_validation.cross_val_score(wrapped_model, X_train, Y_train, scoring="mean_squared_error", cv=splits))
    # print "RandomForestRegression(tuned): {:.3f} +/- {:.3f}".format(scores.mean(), scores.std())

    # BROKEN! GradientBoosting only handles univariate output
    model = VectorRegression(sklearn.ensemble.GradientBoostingRegressor())
    scores = mse_to_rms(sklearn.cross_validation.cross_val_score(model, X_train, Y_train, scoring="mean_squared_error", cv=splits))
    print "GradientBoostingRegressor: {:.3f} +/- {:.3f}".format(scores.mean(), scores.std())


    baseline_model.fit(X_train, Y_train)

    # retrain a model on the full data
    # model = sklearn.linear_model.LinearRegression()
    wrapped_model.fit(X_train, Y_train)

    predict_test_data(wrapped_model, scaler, args, Y_train, baseline_model=baseline_model)


def predict_test_data(model, scaler, args, Y_train, baseline_model=None):
    test_data = load_data(args.testing_dir)
    X_test, Y_test = separate_output(test_data)
    X_test = scaler.transform(X_test)

    test_data[Y_train.columns] = model.predict(X_test)

    if baseline_model:
        baseline_predictions = baseline_model.predict(X_test)
        predictions = model.predict(X_test)

        deltas = numpy.abs(predictions - baseline_predictions) / numpy.abs(baseline_predictions)
        mean_delta = deltas.mean().mean()
        print "Average percent change from baseline predictions: {:.2f}%".format(100. * mean_delta)

        assert mean_delta < 1

    # redo the index as unix timestamp
    test_data.index = test_data.index.astype(numpy.int64) / 10 ** 6
    test_data[Y_test.columns].to_csv(args.prediction_file, index_label="ut_ms")


def separate_output(dataframe):
    dataframe.drop("file_number", axis=1, inplace=True)

    Y = dataframe[[col for col in dataframe.columns if col.startswith("NPWD")]]
    X = dataframe[[col for col in dataframe.columns if not col.startswith("NPWD")]]
    return X, Y


if __name__ == "__main__":
    sys.exit(main())