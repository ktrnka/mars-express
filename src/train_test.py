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
from src import sklearn_helpers
from src.sklearn_helpers import MultivariateRegressionWrapper, print_tuning_scores, mse_to_rms, \
    print_feature_importances


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-analysis", default=False, action="store_true", help="Extra analysis on the data")
    parser.add_argument("--analyse-feature-importance", default=False, action="store_true", help="Analyse feature importance and print them out for some models")
    parser.add_argument("--analyse-hyperparameters", default=False, action="store_true", help="Analyse hyperparameters and print them out for some models")
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

    # time-lagged version
    add_lag_feature(longterm_data, "eclipseduration_min", 2 * 24, data_type=numpy.int64)
    add_lag_feature(longterm_data, "eclipseduration_min", 5 * 24, data_type=numpy.int64)


    # events
    event_data = load_series(find_files(data_dir, "evtf"))
    # merge_umbra_penumbra(data, event_data)
    # fill_events(data, get_event_ranges(event_data, "MAR_UMBRA"), "IN_MAR_UMBRA")
    # fill_events(data, get_event_ranges(event_data, "MRB_/_RANGE_06000KM"), "IN_MRB_/_RANGE_06000KM")
    # fill_events(data, get_event_ranges(event_data, "MSL_/_RANGE_06000KM"), "IN_MSL_/_RANGE_06000KM")

    event_data.drop(["description"], axis=1, inplace=True)
    event_data["event_counts"] = 1
    event_data = event_data.resample("1H").count().reindex(data.index, method="nearest")
    add_lag_feature(event_data, "event_counts", 2, data_type=numpy.int64)
    add_lag_feature(event_data, "event_counts", 5, data_type=numpy.int64)

    dmop_data = load_series(find_files(data_dir, "dmop"))
    dmop_data.drop(["subsystem"], axis=1, inplace=True)
    dmop_data["dmop_counts"] = 1
    dmop_data = dmop_data.resample("1H").count().reindex(data.index, method="nearest")
    add_lag_feature(dmop_data, "dmop_counts", 2, data_type=numpy.int64)
    add_lag_feature(dmop_data, "dmop_counts", 5, data_type=numpy.int64)

    # resample saaf to 30 minute windows before reindexing to smooth it out a bit
    saaf_data = saaf_data.resample("30Min").mean().reindex(data.index, method="nearest")

    longterm_data = longterm_data.reindex(data.index, method="nearest")

    data = pandas.concat([data, saaf_data, longterm_data, dmop_data, event_data], axis=1)
    # print data[data.NPWD2882.isnull()].NPWD2882

    # TODO: Delete any data with missing power info rather than interpolate that (training only, not testing)

    data["days_in_space"] = (data.index - pandas.datetime(year=2003, month=6, day=2)).days

    data.drop(["sx", "sa"], axis=1, inplace=True)
    for feature in ["sz", "sy"]:
        data["sin_{}".format(feature)] = numpy.sin(data[feature] / 360)
        data["cos_{}".format(feature)] = numpy.cos(data[feature] / 360)

    print "Before fillna global:"
    data.info()

    print data.head(100)

    # fix any remaining NaN
    data = data.interpolate().fillna(data.mean())

    return data


def add_lag_feature(data, feature, window, drop=False, data_type=None):
    name = feature + "_rolling_{}h".format(window)
    data[name] = data[feature].rolling(window=window).mean().fillna(method="backfill")

    if data_type:
        data[name] = data[name].astype(data_type)

    if drop:
        data.drop([feature], axis=1, inplace=True)


def compute_upper_bounds(data):
    data = data[[c for c in data.columns if c.startswith("NPWD")]]

    for interval in "7D 1D 12H 6H 2H 1H 30M".split():
        downsampled_data = data.resample(interval).mean()
        upsampled_data = downsampled_data.reindex(data.index, method="pad")

        rms = ((data - upsampled_data) ** 2).mean().mean() ** 0.5
        print "RMS with {} approximation: {:.3f}".format(interval, rms)


def experiment_neural_network(X_train, Y_train, args, splits, tune_params):
    Y_train = Y_train.values
    model = sklearn_helpers.NnRegressor(batch_spec=((1500, -1),), learning_rate=0.01, dropout=0.6, hidden_activation="sigmoid", init="glorot_uniform")
    cross_validate(X_train, Y_train, model, "NnRegressor", splits)

    if args.analyse_hyperparameters and tune_params:
        print "Running hyperparam opt"
        nn_hyperparams = {
            "input_noise": scipy.stats.uniform(0, 0.1),
            "dropout": [0.5],
            "learning_rate": [0.01, 0.005],
            "batch_spec": [((1000, -1),)],
            "hidden_activation": ["sigmoid"]
        }
        model = sklearn_helpers.NnRegressor(batch_spec=((1000, -1),), init="glorot_uniform")
        wrapped_model = sklearn.grid_search.RandomizedSearchCV(model, nn_hyperparams, n_iter=10, n_jobs=1, scoring="mean_squared_error")
        # cross_validate(X_train, Y_train, wrapped_model, "RandomizedSearchCV(NnRegressor)", splits)

        wrapped_model.fit(X_train, Y_train)
        print_tuning_scores(wrapped_model, reverse=True, score_transformer=mse_to_rms)


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
        compute_upper_bounds(train_data)

    scaler = sklearn.preprocessing.RobustScaler()
    feature_names = X_train.columns
    X_train = scaler.fit_transform(X_train)

    # lower bound: predict mean
    baseline_model = sklearn.dummy.DummyRegressor("mean")
    cross_validate(X_train, Y_train, baseline_model, "DummyRegressor(mean)", splits)

    model = sklearn.linear_model.LinearRegression()
    cross_validate(X_train, Y_train, model, "LinearRegression", splits)

    experiment_bagged_linear_regression(X_train, Y_train, args, splits, tune_params=False)

    # model = sklearn.linear_model.Lars()
    # cross_validate(X_train, Y_train, model, "Lars", splits)

    experiment_neural_network(X_train, Y_train, args, splits, tune_params=True)

    # experiment_adaboost(X_train, Y_train, args, feature_names, splits, tune_params=False)

    # experiment_random_forest(X_train, Y_train, args, feature_names, splits, tune_params=False)

    # experiment_gradient_boosting(X_train, Y_train, args, feature_names, splits, tune_params=True)

    if args.prediction_file != "-":
        predict_test_data(X_train, Y_train, scaler, args)


def experiment_bagged_linear_regression(X_train, Y_train, args, splits, tune_params=False):
    model = MultivariateRegressionWrapper(sklearn.ensemble.BaggingRegressor(sklearn.linear_model.LinearRegression(), max_samples=0.9, max_features=12, n_estimators=30))
    cross_validate(X_train, Y_train, model, "Bagging(LinearRegression)", splits)

    if args.analyse_hyperparameters and tune_params:
        bagging_params = {
            "max_samples": scipy.stats.uniform(0.8, 0.2),
            "max_features": scipy.stats.randint(4, X_train.shape[1] + 1)
        }
        base_model = sklearn.ensemble.BaggingRegressor(sklearn.linear_model.LinearRegression(), n_estimators=30)
        model = MultivariateRegressionWrapper(sklearn.grid_search.RandomizedSearchCV(base_model, bagging_params, n_iter=20, n_jobs=1, scoring="mean_squared_error"))
        cross_validate(X_train, Y_train, model, "RandomizedSearchCV(Bagging(LinearRegression))", splits)

        # refit on full data to get a single model and spit out the info
        model.fit(X_train, Y_train)
        model.print_best_params()


def experiment_gradient_boosting(X_train, Y_train, args, feature_names, splits, tune_params=False):
    model = MultivariateRegressionWrapper(sklearn.ensemble.GradientBoostingRegressor(max_features=18, n_estimators=50, learning_rate=0.3, max_depth=4, min_samples_leaf=66))
    cross_validate(X_train, Y_train, model, "GradientBoostingRegressor", splits)

    if args.analyse_hyperparameters and tune_params:
        gb_hyperparams = {
            "learning_rate": scipy.stats.uniform(0.1, 0.5),
            "n_estimators": scipy.stats.randint(20, 100),
            "max_depth": scipy.stats.randint(3, 6),
            "min_samples_leaf": scipy.stats.randint(10, 100),
            # "subsample": [0.9, 1.],
            "max_features": scipy.stats.randint(4, X_train.shape[1] + 1),
            # "init": [None, sklearn_helpers.LinearRegressionWrapper()]
        }
        wrapped_model = MultivariateRegressionWrapper(sklearn.grid_search.RandomizedSearchCV(sklearn.ensemble.GradientBoostingRegressor(), gb_hyperparams, n_iter=20, n_jobs=3, scoring="mean_squared_error"))
        cross_validate(X_train, Y_train, wrapped_model, "RandomizedSearchCV(GradientBoostingRegressor)", splits)

        wrapped_model.fit(X_train, Y_train)
        wrapped_model.print_best_params()

        if args.analyse_feature_importance:
            wrapped_model.print_feature_importances(feature_names)


def experiment_adaboost(X_train, Y_train, args, feature_names, splits, tune_params=False):
    model = MultivariateRegressionWrapper(sklearn.ensemble.AdaBoostRegressor(base_estimator=sklearn.linear_model.LinearRegression(), learning_rate=0.7, loss="square"))
    cross_validate(X_train, Y_train, model, "AdaBoost(LinearRegression)", splits)

    if args.analyse_hyperparameters and tune_params:
        ada_params = {
            "learning_rate": scipy.stats.uniform(0.3, 1.),
            "n_estimators": scipy.stats.randint(20, 100),
            "loss": ["linear", "square", "exponential"]
        }
        model = MultivariateRegressionWrapper(sklearn.grid_search.RandomizedSearchCV(sklearn.ensemble.AdaBoostRegressor(base_estimator=sklearn.linear_model.LinearRegression()), ada_params, scoring="mean_squared_error"))
        cross_validate(X_train, Y_train, model, "RandomSearchCV(AdaBoost(LinearRegression))", splits)

        print "Refitting to show hyperparams"
        model.fit(X_train, Y_train)
        model.print_best_params()


def experiment_random_forest(X_train, Y_train, args, feature_names, splits, tune_params=False):
    # plain model
    model = sklearn.ensemble.RandomForestRegressor(80, min_samples_leaf=30, max_depth=15, max_features=15)
    cross_validate(X_train, Y_train, model, "RandomForestRegressor", splits)

    if args.analyse_hyperparameters and tune_params:
        rf_hyperparams = {
            "min_samples_leaf": scipy.stats.randint(10, 100),
            "max_depth": scipy.stats.randint(5, 15),
            "max_features": scipy.stats.randint(4, X_train.shape[1] + 1),
            "n_estimators": scipy.stats.randint(20, 100)
        }
        wrapped_model = sklearn.grid_search.RandomizedSearchCV(model, rf_hyperparams, n_iter=10, n_jobs=3, scoring="mean_squared_error")
        cross_validate(X_train, Y_train, wrapped_model, "RandomizedSearchCV(RandomForestRegression)", splits)

        model = sklearn.grid_search.RandomizedSearchCV(sklearn.ensemble.RandomForestRegressor(), rf_hyperparams, n_iter=10, n_jobs=3, cv=splits, scoring="mean_squared_error")
        model.fit(X_train, Y_train)
        print_tuning_scores(model, reverse=True, score_transformer=mse_to_rms)

        if args.analyse_feature_importance:
            print_feature_importances(feature_names, model.best_estimator_)


def cross_validate(X_train, Y_train, model, model_name, splits):
    scores = mse_to_rms(sklearn.cross_validation.cross_val_score(model, X_train, Y_train, scoring="mean_squared_error", cv=splits))
    print "{}: {:.4f} +/- {:.4f}".format(model_name, scores.mean(), scores.std())


def predict_test_data(X_train, Y_train, scaler, args):
    # retrain baseline model as a sanity check
    baseline_model = sklearn.dummy.DummyRegressor("mean")
    baseline_model.fit(X_train, Y_train)

    # retrain a model on the full data
    model = sklearn_helpers.NnRegressor(batch_spec=((1500, -1),), learning_rate=0.01, dropout=0.6, hidden_activation="sigmoid", init="glorot_uniform")
    # model = MultivariateRegressionWrapper(sklearn.ensemble.BaggingRegressor(sklearn.linear_model.LinearRegression(), max_samples=0.9, max_features=12, n_estimators=30))
    model.fit(X_train, Y_train.values)

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