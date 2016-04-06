from __future__ import unicode_literals

import argparse
import collections
import sys

import numpy
import os
import pandas
import scipy.optimize
import scipy.stats
import sklearn
import sklearn.cross_validation
import sklearn.dummy
import sklearn.ensemble
import sklearn.grid_search
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.svm
from src.sklearn_helpers import MultivariateRegressionWrapper, print_tuning_scores, mse_to_rms, \
    print_feature_importances


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


def fill_events(data, event_ranges, dest_col, add_duration=False):
    # TODO: Rewrite this to return a Series with the same index
    data[dest_col] = 0

    for event in event_ranges:
        closest_start = data.index.searchsorted(event["start"], side="right")
        closest_end = data.index.searchsorted(event["end"], side="right")

        if not add_duration:
            data.loc[closest_start:closest_end, dest_col] = 1
        else:
            if closest_start < len(data):
                data.loc[closest_start, dest_col] = (data.index[closest_start] - event["start"]).seconds
                data.loc[closest_end, dest_col] = (event["end"] - data.index[closest_start]).seconds


def merge_umbra_penumbra(data, event_data):
    for obj in ["MAR"]:
        for event in "PENUMBRA UMBRA".split():
            prefix = "{}_{}".format(obj, event)
            fill_events(data, get_event_ranges(event_data, prefix), "IN_" + prefix)

            if event == "UMBRA":
                pen_col = "IN_{}_{}".format(obj, "PENUMBRA")
                mean_before = data[pen_col].mean()
                data.loc[data["IN_" + prefix] == 1, pen_col] = 0

                print "Penumbra mean changed from {} to {}".format(mean_before, data[pen_col].mean())

            # add duration columns
            # fill_events(data, get_event_ranges(event_data, prefix), prefix + "_SECONDS_ELAPSED", add_duration=True)


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
    longterm_data = longterm_data.resample("1H").mean().interpolate().fillna(method="backfill")

    # events
    event_data = load_series(find_files(data_dir, "evtf"))
    merge_umbra_penumbra(data, event_data)
    fill_events(data, get_event_ranges(event_data, "MRB_/_RANGE_06000KM"), "IN_MRB_/_RANGE_06000KM")
    fill_events(data, get_event_ranges(event_data, "MSL_/_RANGE_06000KM"), "IN_MSL_/_RANGE_06000KM")

    event_data.drop(["description"], axis=1, inplace=True)
    event_data["event_counts"] = 1
    event_data = event_data.resample("1H").count().reindex(data.index, method="nearest")

    dmop_data = load_series(find_files(data_dir, "dmop"))
    dmop_data.drop(["subsystem"], axis=1, inplace=True)
    dmop_data["dmop_counts"] = 1
    dmop_data = dmop_data.resample("1H").count().reindex(data.index, method="nearest")
    add_lag_feature(dmop_data, "dmop_counts", 24 * 7)

    # resample saaf to 30 minute windows before reindexing to smooth it out a bit
    saaf_data = saaf_data.resample("30Min").mean().reindex(data.index, method="nearest")

    longterm_data = longterm_data.reindex(data.index, method="nearest")

    data = pandas.concat([data, saaf_data, longterm_data, dmop_data, event_data], axis=1)

    data["days_in_space"] = (data.index - pandas.datetime(year=2003, month=6, day=2)).days

    # # replace some features with cut versions
    # for feature in ["sz", "sy", "sx", "dmop_counts_rolling_7d", "dmop_counts", "event_counts"]:
    #     data[feature] = pandas.cut(data[feature], 20, labels=False)

    for feature in ["sz", "sy", "sx", "sa"]:
        data["sin_{}".format(feature)] = numpy.sin(data[feature] / 360)
        data["cos_{}".format(feature)] = numpy.cos(data[feature] / 360)

    # derived angle features
    # for feature in ["sz", "sy"]:
    #     for window in [2, 6, 12]:
    #         data[feature + "_rolling_{}h".format(window)] = data[feature].rolling(window=window).mean().fillna(method="pad")

    # derived long-term features
    # for feature in ["eclipseduration_min", "earthmars_km", "sunmars_km"]:
    #     for window in [24 * 7, 24 * 2, 24 * 14]:
    #         add_lag_feature(data, feature, window)

    # add_lag_feature(data, "eclipseduration_min", 24 * 14)
    # add_lag_feature(data, "sunmars_km", 24 * 14, drop=True)

    return data.fillna(data.mean())


def add_lag_feature(data, feature, window, drop=False):
    data[feature + "_rolling_{}d".format(window / 24)] = data[feature].rolling(window=window).mean().fillna(method="pad")

    if drop:
        data.drop([feature], axis=1, inplace=True)


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

    # # curve fitting (need the pandas DataFrame for the date index
    # time_offsets = TimeSeriesRegressor.get_time_offset(X_train, pandas.datetime(year=2003, month=6, day=2))
    # model = MultivariateRegressionWrapper(TimeSeriesRegressor())
    # cross_validate(time_offsets, Y_train, model, "TimeSeriesRegressor", splits)


    scaler = sklearn.preprocessing.RobustScaler()
    X_train_orig = X_train
    X_train = scaler.fit_transform(X_train)

    # lower bound: predict mean
    baseline_model = sklearn.dummy.DummyRegressor("mean")
    cross_validate(X_train, Y_train, baseline_model, "DummyRegressor(mean)", splits)

    # upper bound: predict average per diem
    if args.extra_analysis:
        compute_upper_bounds(train_data)

    model = sklearn.linear_model.LinearRegression()
    cross_validate(X_train, Y_train, model, "LinearRegression", splits)

    model = MultivariateRegressionWrapper(sklearn.ensemble.BaggingRegressor(sklearn.linear_model.LinearRegression(), max_samples=0.9, max_features=0.8))
    cross_validate(X_train, Y_train, model, "Bagging(LinearRegression)", splits)

    # model = MultivariateRegressionWrapper(sklearn.svm.LinearSVR())
    # cross_validate(X_train, Y_train, model, "LinearSVR", splits)

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

    # if we want tuning scores we need to refit without cross-validation
    model = sklearn.grid_search.RandomizedSearchCV(model, rf_hyperparams, n_iter=10, n_jobs=3, cv=splits)
    model.fit(X_train, Y_train)
    print_tuning_scores(model, reverse=False, score_transformer=mse_to_rms)

    model = MultivariateRegressionWrapper(sklearn.ensemble.GradientBoostingRegressor())
    cross_validate(X_train, Y_train, model, "GradientBoostingRegressor", splits)

    # feature importances on the top N outputs
    scores = collections.Counter({col: Y_train[col].mean() + Y_train[col].std() for col in Y_train.columns})
    feature_importances = collections.defaultdict(list)
    for col, _ in scores.most_common(10):
        print "Feature importances for output {}".format(col)
        model = sklearn.ensemble.GradientBoostingRegressor()
        model.fit(X_train, Y_train[col])

        for feature_name, feature_score in zip(X_train_orig.columns, model.feature_importances_):
            feature_importances[feature_name].append(feature_score)
        print_feature_importances(X_train_orig.columns, model)

    print "\nSummed importances".upper()
    for feature_name, feature_scores in sorted(feature_importances.items(), key=lambda p: sum(p[1]), reverse=True):
        feature_scores = numpy.asarray(feature_scores)
        print "\t{}: {:.3f} +/- {:.3f}".format(feature_name, feature_scores.mean(), feature_scores.std())

    if args.prediction_file != "-":
        baseline_model.fit(X_train, Y_train)

        # retrain a model on the full data
        model = MultivariateRegressionWrapper(sklearn.ensemble.BaggingRegressor(sklearn.linear_model.LinearRegression(), max_samples=0.9, max_features=0.8))
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