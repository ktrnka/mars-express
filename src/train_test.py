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

    X_train = sklearn.preprocessing.StandardScaler().fit_transform(X_train)

    # lower bound: predict mean
    baseline = sklearn.dummy.DummyRegressor("mean")
    baseline_scores = mse_to_rms(sklearn.cross_validation.cross_val_score(baseline, X_train, Y_train, scoring="mean_squared_error", cv=splits))
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


    # model = sklearn.ensemble.RandomForestRegressor(20, min_samples_leaf=100, max_depth=3)
    # scores = mse_to_rms(sklearn.cross_validation.cross_val_score(model, X, Y, scoring="mean_squared_error", cv=splits))
    # print "RandomForestRegression: {:.3f} +/- {:.3f}".format(scores.mean(), scores.std())
    #
    # wrapped_model = sklearn.grid_search.RandomizedSearchCV(model, {"min_samples_leaf": range(10, 100), "max_depth": range(3, 10), "max_features": ["sqrt", "log2", 1.0, 0.3]}, n_iter=20)
    # scores = mse_to_rms(sklearn.cross_validation.cross_val_score(wrapped_model, X, Y, scoring="mean_squared_error", cv=splits))
    # print "RandomForestRegression(tuned): {:.3f} +/- {:.3f}".format(scores.mean(), scores.std())


    # retrain a model on the full data
    baseline = sklearn.dummy.DummyRegressor("mean")
    baseline.fit(X_train, Y_train)

    predict_test_data(baseline, args, Y_train)


def predict_test_data(model, args, Y_train):
    test_data = load_data(args.testing_dir)
    X_test, Y_test = separate_output(test_data)

    test_data[Y_train.columns] = model.predict(X_test)

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