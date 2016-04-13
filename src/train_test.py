from __future__ import unicode_literals

import argparse
import collections
import sys
from operator import itemgetter

import datetime

import numpy
import os
import pandas
import scipy.optimize
import scipy.stats
import logging
import sklearn
import sklearn.cross_validation
import sklearn.decomposition
import sklearn.dummy
import sklearn.ensemble
import sklearn.gaussian_process
import sklearn.grid_search
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
import sklearn.metrics
import sklearn_helpers
from sklearn_helpers import MultivariateRegressionWrapper, mse_to_rms, print_feature_importances
from helpers.multivariate import MultivariateBaggingRegressor
from sklearn.linear_model import LinearRegression


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
    """Date parser for pandas.read_csv"""
    return pandas.to_datetime(column.astype(int), unit="ms")


def find_files(path, substring):
    files = os.listdir(path)
    return sorted(os.path.join(path, f) for f in files if substring in f)


def get_evtf_ranges(event_data, event_prefix):
    """Get time ranges between event_prefix + _START and event_prefix + _END"""
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


def get_evtf_altitude(event_data, index=None):
    """Read the ascend/descend events and compute the current altitude at all points"""
    desc = event_data.description.str.extract(r"(\d+)_KM_DESCEND", expand=False).fillna(0).astype(numpy.int16)
    asc = event_data.description.str.extract(r"(\d+)_KM_ASCEND", expand=False).fillna(0).astype(numpy.int16)

    alt_delta = asc - desc
    alt = alt_delta.cumsum()
    alt -= alt.min()

    if index is not None:
        alt = alt.resample("1H").mean().interpolate().reindex(index, method="nearest")

    return alt

def get_ftl_periods(ftl_slice):
    """Get time ranges for FTL data (first two columns are start and end time so it's simple)"""
    for row in ftl_slice.itertuples():
        yield {"start": row[0], "end": row[1]}


def get_event_series(datetime_index, event_ranges):
    """Create a boolean series showing when in the datetime_index we're in the time ranges in the event_ranges"""
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


def load_data(data_dir, resample_interval=None, filter_null_power=False, derived_features=True):
    # load the base power data
    data = load_series(find_files(data_dir, "power"), add_file_number=True, resample_interval=resample_interval)

    event_sampling_index = pandas.DatetimeIndex(freq="5Min", start=data.index.min(), end=data.index.max())
    event_sampled_df = pandas.DataFrame(index=event_sampling_index)

    ### LTDATA ###
    longterm_data = load_series(find_files(data_dir, "ltdata"))

    # as far as I can tell this doesn't make a difference
    longterm_data = longterm_data.resample("1H").mean().interpolate().fillna(method="backfill")

    # time-lagged version
    add_lag_feature(longterm_data, "eclipseduration_min", 2 * 24, "2d", data_type=numpy.int64)
    add_lag_feature(longterm_data, "eclipseduration_min", 5 * 24, "5d", data_type=numpy.int64)

    ### FTL ###
    ftl_data = load_series(find_files(data_dir, "ftl"), date_cols=["utb_ms", "ute_ms"])

    event_sampled_df["flagcomms"] = get_event_series(event_sampling_index, get_ftl_periods(ftl_data[ftl_data.flagcomms]))
    add_lag_feature(event_sampled_df, "flagcomms", 12, "1h")
    add_lag_feature(event_sampled_df, "flagcomms", 24, "2h")

    # select columns or take preselected ones
    for ftl_type in ["SLEW", "EARTH", "INERTIAL", "D4PNPO", "MAINTENANCE", "NADIR", "WARMUP", "ACROSS_TRACK", "RADIO_SCIENCE"]:
        dest_name = "FTL_" + ftl_type
        event_sampled_df[dest_name] = get_event_series(event_sampled_df.index, get_ftl_periods(ftl_data[ftl_data["type"] == ftl_type]))
        add_lag_feature(event_sampled_df, dest_name, 12, "1h")
        add_lag_feature(event_sampled_df, dest_name, 24, "2h")

    ### EVTF ###
    event_data = load_series(find_files(data_dir, "evtf"))

    for event_name in ["MAR_UMBRA", "MRB_/_RANGE_06000KM", "MSL_/_RANGE_06000KM"]:
        dest_name = "EVTF_IN_" + event_name
        event_sampled_df[dest_name] = get_event_series(event_sampling_index, get_evtf_ranges(event_data, event_name))
        add_lag_feature(event_sampled_df, dest_name, 12, "1h")

    altitude_series = get_evtf_altitude(event_data, index=data.index)
    event_data.drop(["description"], axis=1, inplace=True)
    event_data["EVTF_event_counts"] = 1
    event_data = event_data.resample("1H").count().reindex(data.index, method="nearest")
    event_data["EVTF_altitude"] = altitude_series
    add_lag_feature(event_data, "EVTF_event_counts", 2, "2h", data_type=numpy.int64)
    add_lag_feature(event_data, "EVTF_event_counts", 5, "5h", data_type=numpy.int64)

    ### DMOP ###
    dmop_data = load_series(find_files(data_dir, "dmop"))
    dmop_data.drop(["subsystem"], axis=1, inplace=True)
    dmop_data["DMOP_event_counts"] = 1
    dmop_data = dmop_data.resample("1H").count().reindex(data.index, method="nearest")
    add_lag_feature(dmop_data, "DMOP_event_counts", 2, "2h", data_type=numpy.int64)
    add_lag_feature(dmop_data, "DMOP_event_counts", 5, "5h", data_type=numpy.int64)

    ### SAAF ###
    saaf_data = load_series(find_files(data_dir, "saaf"))
    saaf_data = saaf_data.resample("1H").mean().reindex(data.index, method="nearest").interpolate()

    longterm_data = longterm_data.reindex(data.index, method="nearest")

    data = pandas.concat([data, saaf_data, longterm_data, dmop_data, event_data, event_sampled_df.reindex(data.index, method="nearest")], axis=1)
    # data = pandas.concat([data, longterm_data], axis=1)

    if filter_null_power:
        previous_size = data.shape[0]
        data = data[data.NPWD2532.notnull()]
        if data.shape[0] < previous_size:
            print "Reduced data from {:,} rows to {:,}".format(previous_size, data.shape[0])

    data["days_in_space"] = (data.index - pandas.datetime(year=2003, month=6, day=2)).days

    if derived_features:
        for col in [c for c in data.columns if "EVTF_IN_MRB" in c]:
            add_transformation_feature(data, col, "gradient")
        add_transformation_feature(data, "FTL_EARTH_rolling_1h", "gradient")
        add_transformation_feature(data, "DMOP_event_counts", "log", drop=True)
        add_transformation_feature(data, "DMOP_event_counts_rolling_2h", "gradient", drop=True)
        add_transformation_feature(data, "occultationduration_min", "log", drop=True)
        add_transformation_feature(data, "sy", "log", drop=True)
        add_transformation_feature(data, "sa", "log", drop=True)

    # simple check on NaN
    data_na = data[[c for c in data.columns if not c.startswith("NPWD")]].isnull().sum()
    if data_na.sum() > 0:
        print "Null values in feature matrix:"

        for feature, na_count in data_na.iteritems():
            if na_count > 0:
                print "\t{}: {:.1f}% null ({:,} / {:,})".format(feature, 100. * na_count / len(data), na_count, len(data))

        sys.exit(-1)

    return data


def add_lag_feature(data, feature, window, time_suffix, drop=False, data_type=None):
    name = feature + "_rolling_{}".format(time_suffix)
    data[name] = data[feature].rolling(window=window).mean().fillna(method="backfill")

    if data_type:
        data[name] = data[name].astype(data_type)

    if drop:
        data.drop([feature], axis=1, inplace=True)


def add_transformation_feature(data, feature, transform, drop=False):
    new_name = feature + "_" + transform

    if transform == "log":
        transformed = numpy.log(data[feature] + 1)
    elif transform == "square":
        transformed = numpy.square(data[feature])
    elif transform == "sqrt":
        transformed = numpy.sqrt(data[feature])
    elif transform == "gradient":
        transformed = numpy.gradient(data[feature])
    else:
        print "Unknown transform {} specified".format(transform)
        sys.exit(-1)

    data[new_name] = transformed

    if drop:
        data.drop([feature], axis=1, inplace=True)



def compute_upper_bounds(data):
    data = data[[c for c in data.columns if c.startswith("NPWD")]]

    for interval in "7D 1D 12H 6H 2H 1H 30M".split():
        downsampled_data = data.resample(interval).mean()
        upsampled_data = downsampled_data.reindex(data.index, method="pad")

        rms = ((data - upsampled_data) ** 2).mean().mean() ** 0.5
        print "RMS with {} approximation: {:.3f}".format(interval, rms)

def make_nn():
    """Make a neural network model with reasonable default args"""
    # feature_selector = sklearn.feature_selection.VarianceThreshold(.9 * .1)
    fs = sklearn.feature_selection.SelectFromModel(sklearn.linear_model.LinearRegression(), threshold="0.05*mean", prefit=False)

    # fs = sklearn.feature_selection.RFE(LinearRegression(), n_features_to_select=50)

    scaler = sklearn.preprocessing.StandardScaler()

    model = sklearn_helpers.NnRegressor(num_epochs=500,
                                        batch_size=200,
                                        learning_rate=0.008,
                                        dropout=0.5,
                                        activation="sigmoid",
                                        input_noise=0.05,
                                        hidden_units=100,
                                        early_stopping=True,
                                        loss="mse",
                                        l2=0.0001,
                                        maxnorm=True,
                                        assert_finite=False,
                                        verbose=0)

    # return sklearn.pipeline.Pipeline([("fs", fs), ("nn", model)])

    return model

@sklearn_helpers.Timed
def experiment_neural_network(X_train, Y_train, args, splits, tune_params, use_pca=False):
    Y_train = Y_train.values

    # PCA
    if use_pca:
        original_shape = X_train.shape
        X_train = sklearn.decomposition.PCA(n_components=0.99, whiten=True).fit_transform(X_train)
        print "PCA reduced number of features from {} to {}".format(original_shape[1], X_train.shape[1])

    # X_train = sklearn.preprocessing.StandardScaler().fit_transform(X_train)

    model = make_nn()
    cross_validate(X_train, Y_train, model, "NnRegressor", splits)

    if args.analyse_hyperparameters and tune_params:
        print "Running hyperparam opt"
        nn_hyperparams = {
            "input_noise": sklearn_helpers.RandomizedSearchCV.uniform(0., 0.1),
            "dropout": [0.4, 0.45, 0.5, 0.55, 0.6],
            "learning_rate": sklearn_helpers.RandomizedSearchCV.exponential(0.05, 0.0005),
            "activation": ["sigmoid", "elu", "relu", "tanh"],
            "hidden_units": [25, 50, 75, 100]
        }
        model = make_nn()
        wrapped_model = sklearn_helpers.RandomizedSearchCV(model, nn_hyperparams, n_iter=20, n_jobs=1, scoring="mean_squared_error")
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


def verify_data(train_df, test_df, filename):
    # test stddevs
    train_std = train_df.std()
    for feature, std in train_std.iteritems():
        if std < 0.1:
            print "{} stddev {}".format(feature, std)

    # scale both input and output
    train = sklearn.preprocessing.RobustScaler().fit_transform(train_df)
    test = sklearn.preprocessing.StandardScaler().fit_transform(test_df)

    # find inputs with values over 10x IQR from median
    train_deviants = numpy.abs(train) > 10
    train_deviant_rows = train_deviants.sum(axis=1) > 0

    deviant_df = pandas.DataFrame(numpy.hstack([train[train_deviant_rows], test[train_deviant_rows]]), columns=list(train_df.columns) + list(test_df.columns))
    if deviant_df.shape[0] > 0:
        print "Found {:,} deviant rows, saving to {}".format(deviant_df.shape[0], filename)
        deviant_df.to_csv(filename)
    else:
        print "No deviant rows"

def main():
    args = parse_args()

    train_data = load_data(args.training_dir, resample_interval=args.resample, filter_null_power=True)

    # cross validation by year
    # splits = sklearn_helpers.TimeCV(train_data.shape[0], 10, min_training=0.4, test_splits=3, gap=1)
    # splits = sklearn.cross_validation.KFold(train_data.shape[0], 5, shuffle=True)
    splits = sklearn.cross_validation.LeaveOneLabelOut(train_data["file_number"])

    # just use the biggest one for now
    X_train, Y_train = separate_output(train_data)
    verify_data(X_train, Y_train, "training_deviants.csv")

    if args.extra_analysis:
        X_train.info()
        print X_train.describe()
        compute_upper_bounds(train_data)

    if args.feature_pairs:
        experiment_pairwise_features(X_train, Y_train, splits)

    scaler = sklearn_helpers.ExtraRobustScaler()
    # scaler = sklearn.preprocessing.RobustScaler()

    feature_names = X_train.columns
    X_train = scaler.fit_transform(X_train)

    # lower bound: predict mean
    baseline_model = sklearn.dummy.DummyRegressor("mean")
    cross_validate(X_train, Y_train, baseline_model, "DummyRegressor(mean)", splits)

    model = sklearn.linear_model.LinearRegression()
    cross_validate(X_train, Y_train, model, "LinearRegression", splits)

    experiment_bagged_linear_regression(X_train, Y_train, args, splits, tune_params=True)

    experiment_random_forest(X_train, Y_train, args, feature_names, splits, tune_params=False)

    # experiment_adaboost(X_train, Y_train, args, feature_names, splits, tune_params=False)

    # experiment_gradient_boosting(X_train, Y_train, args, feature_names, splits, tune_params=True)

    experiment_neural_network(X_train, Y_train, args, splits, tune_params=False)

    if args.prediction_file != "-":
        predict_test_data(X_train, Y_train, scaler, args)


def make_blr():
    """Make a bagged linear regression model with reasonable default args"""
    return MultivariateBaggingRegressor(LinearRegression(), max_samples=0.9, max_features=30, n_estimators=30)

@sklearn_helpers.Timed
def experiment_bagged_linear_regression(X_train, Y_train, args, splits, tune_params=False):
    Y_train = Y_train.values

    model = make_blr()
    cross_validate(X_train, Y_train, model, "Bagging(LinearRegression)", splits)

    if args.analyse_hyperparameters and tune_params:
        bagging_params = {
            "max_samples": sklearn_helpers.RandomizedSearchCV.uniform(0.8, 1.),
            "max_features": sklearn_helpers.RandomizedSearchCV.uniform(20, X_train.shape[1])
        }
        base_model = make_blr()
        model = sklearn_helpers.RandomizedSearchCV(base_model, bagging_params, n_iter=20, n_jobs=1, scoring="mean_squared_error")
        # cross_validate(X_train, Y_train, model, "RandomizedSearchCV(Bagging(LinearRegression))", splits)

        # refit on full data to get a single model and spit out the info
        model.fit(X_train, Y_train)
        model.print_tuning_scores()


def make_gb():
    return MultivariateRegressionWrapper(sklearn.ensemble.GradientBoostingRegressor(max_features=30, n_estimators=40, subsample=0.9, learning_rate=0.3, max_depth=4, min_samples_leaf=50))

@sklearn_helpers.Timed
def experiment_gradient_boosting(X_train, Y_train, args, feature_names, splits, tune_params=False):
    model = make_gb()
    cross_validate(X_train, Y_train, model, "GradientBoostingRegressor", splits)

    if args.analyse_hyperparameters and tune_params:
        gb_hyperparams = {
            "learning_rate": scipy.stats.uniform(0.1, 0.5),
            "n_estimators": scipy.stats.randint(20, 50),
            "max_depth": scipy.stats.randint(3, 6),
            "min_samples_leaf": scipy.stats.randint(10, 100),
            "subsample": [0.9, 1.],
            "max_features": scipy.stats.randint(8, X_train.shape[1])
        }
        wrapped_model = MultivariateRegressionWrapper(sklearn.grid_search.RandomizedSearchCV(sklearn.ensemble.GradientBoostingRegressor(), gb_hyperparams, n_iter=20, n_jobs=3, scoring="mean_squared_error"))
        cross_validate(X_train, Y_train, wrapped_model, "RandomizedSearchCV(GradientBoostingRegressor)", splits)

        wrapped_model.fit(X_train, Y_train)
        wrapped_model.print_best_params()

        if args.analyse_feature_importance:
            wrapped_model.print_feature_importances(feature_names)


def experiment_adaboost(X_train, Y_train, args, feature_names, splits, tune_params=False):
    model = MultivariateRegressionWrapper(sklearn.ensemble.AdaBoostRegressor(base_estimator=sklearn.linear_model.LinearRegression(), n_estimators=4, learning_rate=0.5, loss="square"))
    cross_validate(X_train, Y_train, model, "AdaBoost(LinearRegression)", splits)

    if args.analyse_hyperparameters and tune_params:
        ada_params = {
            "learning_rate": scipy.stats.uniform(0.2, 1.),
            "n_estimators": scipy.stats.randint(2, 10)
        }
        base_model = sklearn.ensemble.AdaBoostRegressor(base_estimator=sklearn.linear_model.LinearRegression(), loss="square")
        model = MultivariateRegressionWrapper(sklearn.grid_search.RandomizedSearchCV(base_model, ada_params, scoring="mean_squared_error"))
        cross_validate(X_train, Y_train, model, "RandomSearchCV(AdaBoost(LinearRegression))", splits)

        print "Refitting to show hyperparams"
        model.fit(X_train, Y_train)
        model.print_best_params()


def make_rf():
    """Make a random forest model with reasonable default args"""
    return sklearn.ensemble.RandomForestRegressor(25, min_samples_leaf=100, max_depth=10, max_features=15)


@sklearn_helpers.Timed
def experiment_random_forest(X_train, Y_train, args, feature_names, splits, tune_params=False):
    # plain model
    model = make_rf()
    cross_validate(X_train, Y_train, model, "RandomForestRegressor", splits)

    if args.analyse_hyperparameters and tune_params:
        rf_hyperparams = {
            "min_samples_leaf": scipy.stats.randint(10, 100),
            "max_depth": scipy.stats.randint(5, 15),
            "max_features": scipy.stats.randint(8, X_train.shape[1]),
            "n_estimators": scipy.stats.randint(20, 30)
        }
        wrapped_model = sklearn.grid_search.RandomizedSearchCV(model, rf_hyperparams, n_iter=10, n_jobs=3, scoring="mean_squared_error")
        cross_validate(X_train, Y_train, wrapped_model, "RandomizedSearchCV(RandomForestRegression)", splits)

        model = sklearn_helpers.RandomizedSearchCV(sklearn.ensemble.RandomForestRegressor(), rf_hyperparams, n_iter=10, n_jobs=3, cv=splits, scoring="mean_squared_error")
        model.fit(X_train, Y_train)
        model.print_tuning_scores(score_transformer=mse_to_rms)

        if args.analyse_feature_importance:
            print_feature_importances(feature_names, model.best_estimator_)


def cross_validate(X_train, Y_train, model, model_name, splits, diagnostics=False):
    if diagnostics:
        for i, (train, test) in enumerate(splits):
            # analyse train and test
            print "Split {}".format(i)

            print "\tX[train].mean diff: ", X_train[train].mean(axis=0) - X_train.mean(axis=0)
            print "\tX[train].std diffs: ", X_train[train].std(axis=0) - X_train.std(axis=0)
            print "\tY[train].mean: ", Y_train[train].mean(axis=0)
            print "\tY[train].std: ", Y_train[train].std(axis=0).mean()

            model.fit(X_train[train], Y_train[train])
            predictions = model.predict(X_train[test])
            error = sklearn.metrics.mean_squared_error(Y_train[test], predictions) ** 0.5

            print "\tRMS: {}".format(error)


    scores = mse_to_rms(sklearn.cross_validation.cross_val_score(model, X_train, Y_train, scoring="mean_squared_error", cv=splits))
    print "{}: {:.4f} +/- {:.4f}".format(model_name, scores.mean(), scores.std())


def with_num_features(filename, X):
    return filename.replace(".", ".{}_features.".format(X.shape[1]), 1)

def with_model_name(filename, model):
    return filename.replace(".", ".{}.".format(type(model).__name__), 1)


def with_date(filename):
    return filename.replace(".", ".{}.".format(datetime.datetime.now().strftime("%m_%d")), 1)


def predict_test_data(X_train, Y_train, scaler, args):
    # retrain baseline model as a sanity check
    baseline_model = sklearn.dummy.DummyRegressor("mean")
    baseline_model.fit(X_train, Y_train)

    # retrain a model on the full data
    model = make_nn()
    model.fit(X_train, Y_train.values)

    test_data = load_data(args.testing_dir)
    X_test, Y_test = separate_output(test_data)
    X_test = scaler.transform(X_test)

    test_data[Y_train.columns] = model.predict(X_test)

    verify_predictions(X_test, baseline_model, model)

    # redo the index as unix timestamp
    test_data.index = test_data.index.astype(numpy.int64) / 10 ** 6
    test_data[Y_test.columns].to_csv(with_date(with_model_name(with_num_features(args.prediction_file, X_train), model)), index_label="ut_ms")


def verify_predictions(X_test, baseline_model, model):
    baseline_predictions = baseline_model.predict(X_test)
    predictions = model.predict(X_test)

    deltas = numpy.abs(predictions - baseline_predictions) / numpy.abs(baseline_predictions)
    per_row = deltas.mean()

    unusual_rows = ~(per_row < 5)
    unusual_count = unusual_rows.sum()
    if unusual_count > 0:
        print "{:.1f}% ({:,} / {:,}) of rows have unusual predictions:".format(100. * unusual_count / predictions.shape[0], unusual_count, predictions.shape[0])

        unusual_inputs = X_test[unusual_rows].reshape(-1, X_test.shape[1])
        unusual_outputs = predictions[unusual_rows].reshape(-1, predictions.shape[1])

        for i in xrange(unusual_inputs.shape[0]):
            print "Input: ", unusual_inputs[i]
            print "Output: ", unusual_outputs[i]

    overall_delta = per_row.mean()
    print "Average percent change from baseline predictions: {:.2f}%".format(100. * overall_delta)

    assert overall_delta < 2


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