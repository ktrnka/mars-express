from __future__ import print_function
from __future__ import unicode_literals

import argparse
import collections
import logging
import os
import sys
from operator import itemgetter
from pprint import pprint

import numpy
import pandas
import scipy.optimize
import scipy.stats
import sklearn
import sklearn.cross_validation
import sklearn.decomposition
import sklearn.dummy
import sklearn.ensemble
import sklearn.gaussian_process
import sklearn.grid_search
import sklearn.linear_model
import sklearn.metrics
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
from sklearn.linear_model import LinearRegression

import helpers.general
import helpers.neural
import helpers.sk
from helpers.debug import verify_data
from helpers.features import add_lag_feature, add_transformation_feature, get_event_series, TimeRange
from helpers.sk import print_feature_importances, rms_error


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", default=False, action="store_true", help="Debug logging")

    parser.add_argument("--verify", default=False, action="store_true", help="Run verifications on the input data for outliers and such")

    parser.add_argument("--feature-pairs", default=False, action="store_true", help="Try out pairs of features")
    parser.add_argument("--resample", default="1H", help="Time interval to resample the training data")
    parser.add_argument("--extra-analysis", default=False, action="store_true", help="Extra analysis on the data")
    parser.add_argument("--analyse-feature-importance", default=False, action="store_true", help="Analyse feature importance and print them out for some models")
    parser.add_argument("--analyse-hyperparameters", default=False, action="store_true", help="Analyse hyperparameters and print them out for some models")

    parser.add_argument("training_dir", help="Dir with the training CSV files or joined CSV file with the complete feature matrix")
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
            event_ranges.append(TimeRange(current_start, date))
            current_start = None
    return event_ranges


def get_dmop_subsystem(dmop_data):
    """Extract the subsystem from each record of the dmop data"""
    dmop_subsys = dmop_data.subsystem.str.extract(r"A(?P<subsystem>\w{3}.*)", expand=False)
    dmop_subsys_mapo = dmop_data.subsystem.str.extract(r"(?P<subsystem>.+)\..+", expand=False)

    dmop_subsys.fillna(dmop_subsys_mapo, inplace=True)
    dmop_subsys.fillna(dmop_data.subsystem, inplace=True)

    return dmop_subsys


def get_dmop_ranges(dmop_subsystem, subsystem_name, hours_impact=1.):
    time_offset = pandas.Timedelta(hours=hours_impact)
    for t in dmop_subsystem[dmop_subsystem == subsystem_name].index:
        yield TimeRange(t, t + time_offset)


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
        yield TimeRange(row[0], row[1])


def load_series(files, add_file_number=False, resample_interval=None, date_cols=True):
    data = [pandas.read_csv(f, parse_dates=date_cols, date_parser=parse_dates, index_col=0) for f in files]

    if resample_interval:
        data = [d.resample(resample_interval).mean() for d in data]

    if add_file_number:
        for i, year_data in enumerate(data):
            year_data["file_number"] = i

    data = pandas.concat(data)
    assert isinstance(data, pandas.DataFrame)

    return data


@helpers.general.Timed
def load_data(data_dir, resample_interval=None, filter_null_power=False, derived_features=True):
    logger = helpers.general.get_function_logger()

    if not os.path.isdir(data_dir):
        return pandas.read_csv(data_dir, index_col=0, parse_dates=True)

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

    dmop_subsystems = get_dmop_subsystem(dmop_data)
    # These three were the only ones that had nonzero weights in elastic net when I tried with all over 1000 count
    for subsys in "MMMF10A0 MMMF01A0 ACFE03A".split():
        hours = 0.5
        dest_name = "DMOP_{}_under_{}h_ago".format(subsys, hours)
        event_sampled_df[dest_name] = get_event_series(event_sampling_index, get_dmop_ranges(dmop_subsystems, subsys, hours))
        add_lag_feature(event_sampled_df, dest_name, 12, "1h", drop=True)


    dmop_data.drop(["subsystem"], axis=1, inplace=True)
    dmop_data["DMOP_event_counts"] = 1
    dmop_data = dmop_data.resample("1H").count().reindex(data.index, method="nearest")
    add_lag_feature(dmop_data, "DMOP_event_counts", 2, "2h", data_type=numpy.int64)
    add_lag_feature(dmop_data, "DMOP_event_counts", 5, "5h", data_type=numpy.int64)

    ### SAAF ###
    saaf_data = load_series(find_files(data_dir, "saaf"))
    saaf_data = saaf_data.resample("1H").mean().reindex(data.index, method="nearest").interpolate()

    # best 2 from EN
    for num_days in [1, 8]:
        saaf_data["SAAF_stddev_{}d".format(num_days)] = saaf_data[["sx", "sy", "sz", "sa"]].rolling(num_days * 24).std().fillna(method="bfill").sum(axis=1)

    longterm_data = longterm_data.reindex(data.index, method="nearest")

    data = pandas.concat([data, saaf_data, longterm_data, dmop_data, event_data, event_sampled_df.reindex(data.index, method="nearest")], axis=1)
    assert isinstance(data, pandas.DataFrame)

    if filter_null_power:
        previous_size = data.shape[0]
        data = data[data.NPWD2532.notnull()]
        if data.shape[0] < previous_size:
            logger.info("Reduced data from {:,} rows to {:,}".format(previous_size, data.shape[0]))

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

        # various crazy rolling features
        add_lag_feature(data, "EVTF_IN_MAR_UMBRA_rolling_1h", 50, "50")
        add_lag_feature(data, "EVTF_IN_MRB_/_RANGE_06000KM_rolling_1h", 1600, "1600")
        add_lag_feature(data, "EVTF_event_counts_rolling_5h", 50, "50")
        add_lag_feature(data, "FTL_ACROSS_TRACK", 200, "200")
        add_lag_feature(data, "FTL_NADIR", 400, "400")

    logger.info("DataFrame shape %s", data.shape)
    return data


def compute_upper_bounds(dataframe):
    dataframe = dataframe[[c for c in dataframe.columns if c.startswith("NPWD")]]

    for interval in "7D 1D 12H 6H 2H 1H 30M".split():
        downsampled_data = dataframe.resample(interval).mean()
        upsampled_data = downsampled_data.reindex(dataframe.index, method="pad")

        rms = ((dataframe - upsampled_data) ** 2).mean().mean() ** 0.5
        print("RMS with {} approximation: {:.3f}".format(interval, rms))


def make_nn(history_file=None):
    """Make a plain neural network with reasonable default args"""

    model = helpers.neural.NnRegressor(num_epochs=500,
                                       batch_size=256,
                                       learning_rate=0.004,
                                       dropout=0.5,
                                       activation="elu",
                                       input_noise=0.1,
                                       input_dropout=0.02,
                                       hidden_units=200,
                                       early_stopping=True,
                                       val=0.1,
                                       l2=0.0001,
                                       maxnorm=True,
                                       history_file=history_file,
                                       lr_decay=0.99,
                                       assert_finite=False)

    return model

@helpers.general.Timed
def experiment_neural_network(dataset, tune_params=False):
    model = make_nn()
    cross_validate(dataset, model)

    if tune_params:
        print("Running hyperparam opt")
        nn_hyperparams = {
            "learning_rate": helpers.sk.RandomizedSearchCV.exponential(1e-2, 1e-4),
            "lr_decay": helpers.sk.RandomizedSearchCV.exponential(1 - 1e-2, 1 - 1e-5),
            "input_dropout": helpers.sk.RandomizedSearchCV.uniform(0., 0.1),
            "input_noise": helpers.sk.RandomizedSearchCV.uniform(0.05, 0.2),
            "hidden_units": helpers.sk.RandomizedSearchCV.uniform(100, 500),
            "dropout": helpers.sk.RandomizedSearchCV.uniform(0.3, 0.7)
        }
        model = make_nn()
        model.history_file = None
        wrapped_model = helpers.sk.RandomizedSearchCV(model, nn_hyperparams, n_iter=20, scoring=rms_error, cv=dataset.splits, refit=False)
        # cross_validate(X_train, Y_train, wrapped_model, "RandomizedSearchCV(NnRegressor)", splits)

        wrapped_model.fit(dataset.inputs, dataset.outputs)
        wrapped_model.print_tuning_scores()


def make_rnn(history_file=None, augment_output=False):
    """Make a recurrent neural network with reasonable default args for this task"""
    model = helpers.neural.RnnRegressor(learning_rate=2e-3,
                                        num_units=50,
                                        time_steps=4,
                                        batch_size=256,
                                        num_epochs=1000,
                                        verbose=0,
                                        input_noise=0.1,
                                        input_dropout=0.02,
                                        early_stopping=True,
                                        recurrent_dropout=0.5,
                                        dropout=0.5,
                                        val=0.1,
                                        assert_finite=False,
                                        pretrain=True,
                                        history_file=history_file)

    if augment_output:
        model = helpers.sk.OutputTransformation(model, helpers.sk.QuickTransform.make_append_mean())

    return model


@helpers.general.Timed
def experiment_rnn(dataset, tune_params=False):
    model = make_rnn()
    cross_validate(dataset, model)

    if tune_params:
        hyperparams = {
            "learning_rate": helpers.sk.RandomizedSearchCV.uniform(5e-3, 5e-4),
            "lr_decay": [0.999, 1]
            # "num_units": [25, 50, 100, 200],
            # "dropout": [0.5, 0.55, 0.6],
            # "recurrent_dropout": [0.4, 0.5, 0.6],
            # "learning_rate": helpers.sk.RandomizedSearchCV.exponential(1e-2, 5e-4),
            # "time_steps": [4, 8, 16],
            # "input_dropout": [0.02, 0.04],
        }
        wrapped_model = helpers.sk.RandomizedSearchCV(model, hyperparams, n_iter=3, n_jobs=1, scoring=rms_error, refit=False, cv=dataset.splits)

        wrapped_model.fit(dataset.inputs, dataset.outputs)
        wrapped_model.print_tuning_scores()


def score_feature(X_train, Y_train, splits):
    scaler = sklearn.preprocessing.RobustScaler()
    X_train = scaler.fit_transform(X_train.values.reshape(-1, 1))

    model = sklearn.linear_model.LinearRegression()
    return sklearn.cross_validation.cross_val_score(model, X_train, Y_train, scoring=rms_error, cv=splits).mean()


def save_pairwise_score(name, X, Y, splits, threshold_score, feature_scores):
    score = score_feature(X, Y, splits)

    # only log 5% improvement or more
    if (threshold_score - score) / threshold_score > 0.05:
        feature_scores[name] = score


def experiment_pairwise_features(X_train, Y_train, splits):
    # TODO: redo this as a single ElasticNet
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

    print("Feature correlations")
    for feature, mse in sorted(feature_scores.items(), key=itemgetter(1)):
        print("\t{}: {:.4f}".format(feature, mse))


def experiment_learning_rate_schedule(dataset):
    model = make_nn()
    model.val = 0.1

    # base = no schedule
    print("Baseline NN")
    model.history_file = "nn_default.csv"
    model.fit(dataset.inputs, dataset.outputs)

    # higher init, decay set to reach the same at 40 epochs
    model.schedule = helpers.neural.make_learning_rate_schedule(model.learning_rate, exponential_decay=0.94406087628)
    print("NN with decay")
    model.history_file = "nn_decay.csv"
    model.fit(dataset.inputs, dataset.outputs)

    model.schedule = None
    model.extra_callback = helpers.neural.AdaptiveLearningRateScheduler(model.learning_rate, monitor="val_loss", scale=1.1, window=5)
    print("NN with variance schedule")
    model.history_file = "nn_variance_schedule.csv"
    model.fit(dataset.inputs, dataset.outputs)

    sys.exit(0)


def experiment_output_augmentation(dataset):
    model = make_nn()

    cross_validate(dataset, model)
    cross_validate(dataset, helpers.sk.OutputTransformation(model, helpers.sk.QuickTransform.make_append_mean()))

    sys.exit(0)


def experiment_rnn_elu(dataset):
    model = helpers.neural.RnnRegressor(learning_rate=1e-3,
                                        num_units=50,
                                        time_steps=3,
                                        batch_size=64,
                                        num_epochs=500,
                                        verbose=0,
                                        input_noise=0.1,
                                        input_dropout=0.02,
                                        early_stopping=True,
                                        recurrent_dropout=0.5,
                                        dropout=0.5,
                                        val=0.1,
                                        assert_finite=False,
                                        activation="elu",
                                        pretrain=True)

    print("RNN with followup ELU layer")
    hyperparams = {
        "hidden_layer_sizes": [(50,), (75,), (100,), (200,)],
        "dropout": [0.25, 0.5, .75]
    }

    wrapped_model = helpers.sk.RandomizedSearchCV(model, hyperparams, n_iter=10, n_jobs=1, scoring=rms_error, cv=dataset.splits, refit=False)

    wrapped_model.fit(dataset.inputs, dataset.outputs)
    wrapped_model.print_tuning_scores()

    sys.exit(0)


def experiment_rnn_longer(dataset):
    model = make_rnn()

    time = 10
    print("RNN time ", time)
    model.time_steps = time
    model.num_epochs = 1000
    cross_validate(dataset, model)

    sys.exit(0)


def experiment_rnn_stateful(dataset):
    model = make_rnn()
    model.batch_size = 4
    model.time_steps = 4
    model.val = 0
    # model.verbose = 1
    model.pretrain = False
    model.early_stopping = False
    model.num_epochs = 300

    print("RNN baseline")
    cross_validate(dataset, model)

    print("RNN stateful")
    model.stateful = True
    cross_validate(dataset, model)

    sys.exit(0)


def main():
    args = parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    train_data = load_data(args.training_dir, resample_interval=args.resample, filter_null_power=True)

    # cross validation by year
    # splits = helpers.sk.TimeCV(train_data.shape[0], 10, min_training=0.4, test_splits=3)
    # splits = sklearn.cross_validation.KFold(train_data.shape[0], 7, shuffle=False)
    splits = sklearn.cross_validation.LeaveOneLabelOut(train_data["file_number"])

    # just use the biggest one for now
    X_train, Y_train = separate_output(train_data)

    if args.verify:
        verify_data(X_train, Y_train, None)

    if args.extra_analysis:
        X_train.info()
        print(X_train.describe())
        compute_upper_bounds(train_data)

    if args.feature_pairs:
        experiment_pairwise_features(X_train, Y_train, splits)

    scaler = make_scaler()

    feature_names = X_train.columns
    X_train = scaler.fit_transform(X_train)

    Y_train = Y_train.values

    dataset = helpers.general.DataSet(X_train, Y_train, splits, feature_names)

    baseline_model = sklearn.dummy.DummyRegressor("mean")
    cross_validate(dataset, baseline_model)

    model = sklearn.linear_model.LinearRegression()
    cross_validate(dataset, model)

    experiment_elastic_net(dataset, feature_importance=False)

    experiment_neural_network(dataset, tune_params=False and args.analyse_hyperparameters)

    experiment_rnn(dataset, tune_params=True and args.analyse_hyperparameters)


def experiment_elastic_net(dataset, feature_importance=True):
    model = sklearn.linear_model.ElasticNet(0.01)
    cross_validate(dataset, model)

    if feature_importance:
        model.fit(dataset.inputs, dataset.outputs)

        feature_importances = collections.Counter()
        for fname, fweight in zip(dataset.feature_names, helpers.sk.get_lr_importances(model)):
            feature_importances[fname] = fweight
        print("Feature potentials from ElasticNet (max of abs per-output coefs)")
        pprint(feature_importances.most_common())


def make_scaler():
    pipe = [("scaler", helpers.sk.ClippedRobustScaler())]

    preprocessing_pipeline = sklearn.pipeline.Pipeline(pipe)
    return preprocessing_pipeline


def make_blr(**kwargs):
    """Make a bagged linear regression model with reasonable default args"""
    model = helpers.sk.MultivariateBaggingRegressor(LinearRegression(), max_samples=0.98, max_features=.8, n_estimators=30, **kwargs)
    return model


@helpers.general.Timed
def experiment_bagged_linear_regression(dataset, tune_params=False):
    model = make_blr()
    cross_validate(dataset, model)

    if tune_params:
        bagging_params = {
            "max_samples": helpers.sk.RandomizedSearchCV.uniform(0.85, 1.),
            "max_features": helpers.sk.RandomizedSearchCV.uniform(0.5, 1.)
        }
        base_model = make_blr()
        model = helpers.sk.RandomizedSearchCV(base_model, bagging_params, n_iter=20, n_jobs=1, cv=dataset.splits, scoring=rms_error)

        model.fit(dataset.inputs, dataset.outputs)
        model.print_tuning_scores()


@helpers.general.Timed
def experiment_output_transform(dataset):
    base_model = make_rnn()
    cross_validate(dataset, base_model)

    # plain mean
    print("With mean of outputs")
    output_transformer = helpers.sk.QuickTransform.make_append_mean()
    model = helpers.sk.OutputTransformation(base_model, output_transformer)
    cross_validate(dataset, model)

    # with time-delayed mean
    print("With time-delay of 7")
    output_transformer = helpers.sk.QuickTransform.make_append_rolling(7)
    model = helpers.sk.OutputTransformation(base_model, output_transformer)
    cross_validate(dataset, model)

    sys.exit(0)


def make_rf():
    """Make a random forest model with reasonable default args"""
    return sklearn.ensemble.RandomForestRegressor(25, min_samples_leaf=35, max_depth=34, max_features=20)


@helpers.general.Timed
def experiment_random_forest(X_train, Y_train, args, feature_names, splits, tune_params=False):
    model = make_rf()
    cross_validate(X_train, Y_train, model, splits)

    if args.analyse_hyperparameters and tune_params:
        rf_hyperparams = {
            "min_samples_leaf": scipy.stats.randint(10, 100),
            "max_depth": scipy.stats.randint(5, X_train.shape[1]),
            "max_features": scipy.stats.randint(8, X_train.shape[1]),
            "n_estimators": scipy.stats.randint(20, 30)
        }
        wrapped_model = helpers.sk.RandomizedSearchCV(model, rf_hyperparams, n_iter=10, n_jobs=3, scoring=rms_error)
        wrapped_model.fit(X_train, Y_train)
        wrapped_model.print_tuning_scores()

        if args.analyse_feature_importance:
            print_feature_importances(feature_names, model.best_estimator_)


def cross_validate(dataset, model, n_jobs=1):
    scores = sklearn.cross_validation.cross_val_score(model, dataset.inputs, dataset.outputs, scoring=rms_error, cv=dataset.splits, n_jobs=n_jobs)
    print("{}: {:.4f} +/- {:.4f}".format(helpers.sk.get_model_name(model), -scores.mean(), scores.std()))


def separate_output(df, num_outputs=None):
    logger = helpers.general.get_function_logger()
    df = df.drop("file_number", axis=1)

    Y = df[[col for col in df.columns if col.startswith("NPWD")]]
    if num_outputs:
        scores = collections.Counter({col: Y[col].mean() + Y[col].std() for col in Y.columns})
        Y = Y[[col for col, _ in scores.most_common(num_outputs)]]

    X = df[[col for col in df.columns if not col.startswith("NPWD")]]
    logger.info("X, Y shapes %s %s", X.shape, Y.shape)
    return X, Y


if __name__ == "__main__":
    sys.exit(main())