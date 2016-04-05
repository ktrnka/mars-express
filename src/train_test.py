from __future__ import unicode_literals
import sys
import argparse

import collections
from operator import itemgetter

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


def get_event_ranges(event_data, event_prefix):
    current_start = None
    event_ranges = []
    for date, row in event_data[event_data.description.str.startswith(event_prefix)].iterrows():
        if row["description"].endswith("_START"):
            current_start = date
        elif current_start:
            assert row["description"].endswith("_END")
            event_ranges.append({"duration": date - current_start,
                                 "start": current_start,
                                 "end": date})
            current_start = None
    return event_ranges


def fill_events(data, event_data, prefix):
    col = "IN_" + prefix
    data[col] = 0

    for event in get_event_ranges(event_data, prefix):
        closest_start = data.index.searchsorted(event["start"], side="right")
        closest_end = data.index.searchsorted(event["end"], side="left")
        data.loc[closest_start:closest_end, col] = 1


def merge_umbra_penumbra(data, event_data):
    for obj in "MAR PHO DEI".split():
        for event in "UMBRA PENUMBRA".split():
            fill_events(data, event_data, "{}_{}".format(obj, event))


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

    # as far as I can tell this doesn't make a difference
    # longterm_data = longterm_data.resample("1H").mean().interpolate().fillna(method="backfill")

    # events
    # event_data = load_series(find_files(data_dir, "evtf"))
    # merge_umbra_penumbra(data, event_data)

    # dmop_data = load_series(find_files(data_dir, "dmop"))
    # dmop_data["subsystem"] = dmop_data.subsystem.str.replace(r"\..+", "")
    # dmop_data["dummy"] = 1
    # dmop_data = dmop_data.pivot_table(index=dmop_data.index, columns="subsystem", values="dummy").resample("1H").count()

    # dmop_data = dmop_data.reindex(data.index, method="nearest")
    saaf_data = saaf_data.reindex(data.index, method="nearest")
    longterm_data = longterm_data.reindex(data.index, method="nearest")

    data = pandas.concat([data, saaf_data, longterm_data], axis=1)

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

def print_tuning_scores(tuned_estimator, reverse=True):
    """Show the cross-validation scores and hyperparamters from a grid or random search"""
    for test in sorted(tuned_estimator.grid_scores_, key=itemgetter(1), reverse=reverse):
        print "Validation score {:.2f} +/- {:.2f}, Hyperparams {}".format(100. * test.mean_validation_score,
                                                                          100. * test.cv_validation_scores.std(),
                                                                          test.parameters)

def main():
    args = parse_args()

    train_data = load_data(args.training_dir, resample_interval="1H")

    # cross validation by year
    splits = sklearn.cross_validation.LeaveOneLabelOut(train_data["file_number"])

    # just use the biggest one for now
    X_train, Y_train = separate_output(train_data, num_outputs=3)

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

    if args.extra_analysis:
        model = MultivariateRegressionWrapper(sklearn.linear_model.LinearRegression())
        wrapper_scores = mse_to_rms(sklearn.cross_validation.cross_val_score(model, X_train, Y_train, scoring="mean_squared_error", cv=splits))
        print "\tMultivariate wrapper change on LinearRegression: {}".format((wrapper_scores - scores).mean())

    model = MultivariateRegressionWrapper(sklearn.ensemble.BaggingRegressor(sklearn.linear_model.LinearRegression(), max_samples=0.9, max_features=0.5))
    cross_validate(X_train, Y_train, model, "Bagging(LinearRegression)", splits)

    model = sklearn.ensemble.RandomForestRegressor(20, min_samples_leaf=100, max_depth=3)
    cross_validate(X_train, Y_train, model, "RandomForestRegressor", splits)

    rf_hyperparams = {
        "min_samples_leaf": scipy.stats.randint(10, 100),
        "max_depth": scipy.stats.randint(3, 10),
        "max_features": ["sqrt", "log2", 1.0, 0.3],
        "n_estimators": scipy.stats.randint(20, 100)
    }
    wrapped_model = sklearn.grid_search.RandomizedSearchCV(model, rf_hyperparams, n_iter=10, n_jobs=3)
    cross_validate(X_train, Y_train, wrapped_model, "RandomizedSearchCV(RandomForestRegression)", splits)

    model = MultivariateRegressionWrapper(sklearn.ensemble.GradientBoostingRegressor())
    cross_validate(X_train, Y_train, model, "GradientBoostingRegressor", splits)

    if args.prediction_file != "-":
        baseline_model.fit(X_train, Y_train)

        # retrain a model on the full data
        # model = sklearn.linear_model.LinearRegression()
        model.fit(X_train, Y_train)

        predict_test_data(model, scaler, args, Y_train, baseline_model=baseline_model)


def cross_validate(X_train, Y_train, model, model_name, splits):
    scores = mse_to_rms(
        sklearn.cross_validation.cross_val_score(model, X_train, Y_train, scoring="mean_squared_error", cv=splits))
    print "{}: {:.3f} +/- {:.3f}".format(model_name, scores.mean(), scores.std())


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


def separate_output(dataframe, num_outputs=None):
    dataframe.drop("file_number", axis=1, inplace=True)

    Y = dataframe[[col for col in dataframe.columns if col.startswith("NPWD")]]
    if num_outputs:
        scores = collections.Counter({col: Y[col].mean() + Y[col].std() for col in Y.columns})
        Y = Y[[col for col, _ in scores.most_common(num_outputs)]]

    X = dataframe[[col for col in dataframe.columns if not col.startswith("NPWD")]]
    return X, Y


if __name__ == "__main__":
    sys.exit(main())