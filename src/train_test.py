from __future__ import unicode_literals

import argparse
import collections
import sys
from operator import itemgetter

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
import sklearn.gaussian_process
from src import sklearn_helpers
from src.sklearn_helpers import MultivariateRegressionWrapper, print_tuning_scores, mse_to_rms, \
    print_feature_importances


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-pairs", default=False, action="store_true", help="Try out pairs of features")
    parser.add_argument("--resample", default="1H", help="Time interval to resample the training data")
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


def get_evtf_ranges(event_data, event_prefix):
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


def get_ftl_periods(ftl_slice):
    for row in ftl_slice.itertuples():
        yield {"start": row[0], "end": row[1]}


def get_event_series(datetime_index, event_ranges):
    series = pandas.Series(data=0, index=datetime_index, dtype=numpy.int8)

    for event in event_ranges:
        closest_start = series.index.searchsorted(event["start"], side="right")
        closest_end = series.index.searchsorted(event["end"], side="right")
        series.loc[closest_start:closest_end] = 1

    return series


def load_series(files, add_file_number=False, resample_interval=None, date_cols=True):
    data = [pandas.read_csv(f, parse_dates=date_cols, date_parser=parse_dates, index_col=0) for f in files]

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
    add_lag_feature(longterm_data, "eclipseduration_min", 2 * 24, "2d", data_type=numpy.int64)
    add_lag_feature(longterm_data, "eclipseduration_min", 5 * 24, "5d", data_type=numpy.int64)

    event_sampling_index = pandas.DatetimeIndex(freq="5Min", start=data.index.min(), end=data.index.max())

    # ftl
    ftl_data = load_series(find_files(data_dir, "ftl"), date_cols=["utb_ms", "ute_ms"])

    event_sampled_df = pandas.DataFrame(index=event_sampling_index)
    event_sampled_df["flagcomms"] = get_event_series(event_sampling_index, get_ftl_periods(ftl_data[ftl_data.flagcomms]))
    add_lag_feature(event_sampled_df, "flagcomms", 12, "1h")
    add_lag_feature(event_sampled_df, "flagcomms", 24, "2h")

    for ftl_type, count in ftl_data["type"].value_counts().iteritems():
        if count > 100:
            dest_name = "FTL_" + ftl_type
            event_sampled_df[dest_name] = get_event_series(event_sampled_df.index, get_ftl_periods(ftl_data[ftl_data["type"] == ftl_type]))
            add_lag_feature(event_sampled_df, dest_name, 12, "1h")
            add_lag_feature(event_sampled_df, dest_name, 24, "2h")


    # events
    event_data = load_series(find_files(data_dir, "evtf"))

    for event_name in ["MAR_UMBRA", "MRB_/_RANGE_06000KM", "MSL_/_RANGE_06000KM"]:
        dest_name = "EVTF_IN_" + event_name
        event_sampled_df[dest_name] = get_event_series(event_sampling_index, get_evtf_ranges(event_data, event_name))
        add_lag_feature(event_sampled_df, dest_name, 12, "1h")

    event_data.drop(["description"], axis=1, inplace=True)
    event_data["event_counts"] = 1
    event_data = event_data.resample("1H").count().reindex(data.index, method="nearest")
    add_lag_feature(event_data, "event_counts", 2, "2h", data_type=numpy.int64)
    add_lag_feature(event_data, "event_counts", 5, "5h", data_type=numpy.int64)

    dmop_data = load_series(find_files(data_dir, "dmop"))
    dmop_data.drop(["subsystem"], axis=1, inplace=True)
    dmop_data["dmop_counts"] = 1
    dmop_data = dmop_data.resample("1H").count().reindex(data.index, method="nearest")
    add_lag_feature(dmop_data, "dmop_counts", 2, "2h", data_type=numpy.int64)
    add_lag_feature(dmop_data, "dmop_counts", 5, "2h", data_type=numpy.int64)

    # resample saaf to 30 minute windows before reindexing to smooth it out a bit
    saaf_data = saaf_data.resample("30Min").mean().reindex(data.index, method="nearest")

    longterm_data = longterm_data.reindex(data.index, method="nearest")

    data = pandas.concat([data, saaf_data, longterm_data, dmop_data, event_data, event_sampled_df.reindex(data.index, method="nearest")], axis=1)

    previous_size = data.shape[0]
    data = data[data.NPWD2532.notnull()]
    if data.shape[0] < previous_size:
        print "Reduced data from {:,} rows to {:,}".format(previous_size, data.shape[0])

    data["days_in_space"] = (data.index - pandas.datetime(year=2003, month=6, day=2)).days

    data.drop(["sx", "sa"], axis=1, inplace=True)
    for feature in ["sz", "sy"]:
        data["sin_{}".format(feature)] = numpy.sin(data[feature] / 360)
        data["cos_{}".format(feature)] = numpy.cos(data[feature] / 360)

    # experiments on 7-day data suggest that this might be useful
    data["sunmars_km * days_in_space"] = data["sunmars_km"] * data["days_in_space"]

    # print "Before fillna global:"
    # data.info()
    #
    # print data.head(100)

    # fix any remaining NaN
    data = data.interpolate().fillna(data.mean())

    return data


def add_lag_feature(data, feature, window, time_suffix, drop=False, data_type=None):
    name = feature + "_rolling_{}".format(time_suffix)
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
            "input_noise": sklearn_helpers.RandomizedSearchCV.uniform(0., 0.1),
            "dropout": [0.5],
            "learning_rate": sklearn_helpers.RandomizedSearchCV.exponential(0.01, 0.001),
            "batch_spec": [((1000, -1),)],
            "hidden_activation": ["sigmoid"]
        }
        model = sklearn_helpers.NnRegressor(batch_spec=((1000, -1),), init="glorot_uniform")
        wrapped_model = sklearn_helpers.RandomizedSearchCV(model, nn_hyperparams, n_iter=10, n_jobs=1, scoring="mean_squared_error")
        # cross_validate(X_train, Y_train, wrapped_model, "RandomizedSearchCV(NnRegressor)", splits)

        wrapped_model.fit(X_train, Y_train)
        wrapped_model.print_tuning_scores(score_transformer=mse_to_rms)


def score_feature(X_train, Y_train, splits):
    scaler = sklearn.preprocessing.RobustScaler()
    X_train = scaler.fit_transform(X_train.values.reshape(-1, 1))

    model = sklearn.linear_model.LinearRegression()
    return mse_to_rms(sklearn.cross_validation.cross_val_score(model, X_train, Y_train, scoring="mean_squared_error", cv=splits)).mean()

def save_pairwise_score(name, X, Y, splits, threshold_score, feature_scores):
    score = score_feature(X, Y, splits)

    # only log 5% improvement or more
    if (threshold_score - score) / threshold_score > 0.05:
        feature_scores[name] = score

def experiment_pairwise_features(X_train, Y_train, splits):
    # assume that they're unscaled
    feature_scores = collections.Counter()

    # indep feature scores
    for a in X_train.columns:
        feature_scores[a] = score_feature(X_train[a], Y_train, splits)

    # pairwise feature scores
    for i, a in enumerate(X_train.columns):
        for b in X_train.columns[i:]:
            threshold_score = min(feature_scores[a], feature_scores[b])

            save_pairwise_score("{} * {}".format(a, b), X_train[a] * X_train[b], Y_train, splits, threshold_score, feature_scores)

            if a != b:
                if sum(X_train[b] == 0) == 0:
                    save_pairwise_score("{} / {}".format(a, b), X_train[a] / X_train[b], Y_train, splits, threshold_score, feature_scores)
            save_pairwise_score("{} + {}".format(a, b), X_train[a] + X_train[b], Y_train, splits, threshold_score, feature_scores)
            save_pairwise_score("{} - {}".format(a, b), X_train[a] - X_train[b], Y_train, splits, threshold_score, feature_scores)

    print "Feature correlations"
    for feature, mse in sorted(feature_scores.iteritems(), key=itemgetter(1)):
        print "\t{}: {:.4f}".format(feature, mse)


def main():
    args = parse_args()

    train_data = load_data(args.training_dir, resample_interval=args.resample)

    # cross validation by year
    splits = sklearn.cross_validation.LeaveOneLabelOut(train_data["file_number"])

    # just use the biggest one for now
    X_train, Y_train = separate_output(train_data)

    if args.extra_analysis:
        X_train.info()
        print X_train.describe()
        compute_upper_bounds(train_data)

    if args.feature_pairs:
        experiment_pairwise_features(X_train, Y_train, splits)

    scaler = sklearn.preprocessing.RobustScaler()
    feature_names = X_train.columns
    X_train = scaler.fit_transform(X_train)

    # lower bound: predict mean
    baseline_model = sklearn.dummy.DummyRegressor("mean")
    cross_validate(X_train, Y_train, baseline_model, "DummyRegressor(mean)", splits)

    model = sklearn.linear_model.LinearRegression()
    cross_validate(X_train, Y_train, model, "LinearRegression", splits)

    experiment_bagged_linear_regression(X_train, Y_train, args, splits, tune_params=False)

    # gaussian process (first test, params from their example)
    # model = sklearn.gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
    # cross_validate(X_train, Y_train, model, "GaussianProcess", splits)

    # model = sklearn.linear_model.Lars()
    # cross_validate(X_train, Y_train, model, "Lars", splits)

    experiment_random_forest(X_train, Y_train, args, feature_names, splits, tune_params=False)

    experiment_neural_network(X_train, Y_train, args, splits, tune_params=False)

    # experiment_adaboost(X_train, Y_train, args, feature_names, splits, tune_params=False)

    # experiment_gradient_boosting(X_train, Y_train, args, feature_names, splits, tune_params=True)

    if args.prediction_file != "-":
        predict_test_data(X_train, Y_train, scaler, args)


def experiment_bagged_linear_regression(X_train, Y_train, args, splits, tune_params=False):
    model = MultivariateRegressionWrapper(sklearn.ensemble.BaggingRegressor(sklearn.linear_model.LinearRegression(), max_samples=0.9, max_features=30, n_estimators=30))
    cross_validate(X_train, Y_train, model, "Bagging(LinearRegression)", splits)

    if args.analyse_hyperparameters and tune_params:
        bagging_params = {
            "max_samples": sklearn_helpers.RandomizedSearchCV.uniform(0.8, 1.),
            "max_features": sklearn_helpers.RandomizedSearchCV.uniform(8, X_train.shape[1] + 1)
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
            "n_estimators": scipy.stats.randint(20, 50),
            "max_depth": scipy.stats.randint(3, 6),
            "min_samples_leaf": scipy.stats.randint(10, 100),
            "subsample": [0.9, 1.],
            "max_features": scipy.stats.randint(4, X_train.shape[1] + 1)
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
    model = sklearn.ensemble.RandomForestRegressor(80, min_samples_leaf=30, max_depth=15, max_features=35)
    cross_validate(X_train, Y_train, model, "RandomForestRegressor", splits)

    if args.analyse_hyperparameters and tune_params:
        rf_hyperparams = {
            "min_samples_leaf": scipy.stats.randint(10, 100),
            "max_depth": scipy.stats.randint(5, 15),
            "max_features": scipy.stats.randint(8, X_train.shape[1] + 1),
            "n_estimators": scipy.stats.randint(20, 50)
        }
        wrapped_model = sklearn.grid_search.RandomizedSearchCV(model, rf_hyperparams, n_iter=10, n_jobs=3, scoring="mean_squared_error")
        cross_validate(X_train, Y_train, wrapped_model, "RandomizedSearchCV(RandomForestRegression)", splits)

        model = sklearn_helpers.RandomizedSearchCV(sklearn.ensemble.RandomForestRegressor(), rf_hyperparams, n_iter=10, n_jobs=3, cv=splits, scoring="mean_squared_error")
        model.fit(X_train, Y_train)
        model.print_tuning_scores(score_transformer=mse_to_rms)

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