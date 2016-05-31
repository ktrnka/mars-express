from __future__ import print_function
from __future__ import unicode_literals

from train_test import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", default=False, action="store_true", help="Run verifications on the input data for outliers and such")
    parser.add_argument("--resample", default="1H", help="Time interval to resample the training data")
    parser.add_argument("--extra-analysis", default=False, action="store_true", help="Extra analysis on the data")
    parser.add_argument("num_features", default=40, type=int, help="Number of features to select")
    parser.add_argument("training_dir", help="Dir with the training CSV files or joined CSV file with the complete feature matrix")
    return parser.parse_args()

@helpers.general.Timed
def load_inflated_data(data_dir, resample_interval=None, filter_null_power=False, derived_features=True):
    logger = helpers.general.get_function_logger()

    if not os.path.isdir(data_dir):
        return pandas.read_csv(data_dir, index_col=0, parse_dates=True)

    # load the base power data
    data = load_series(find_files(data_dir, "power"), add_file_number=True, resample_interval=resample_interval)

    event_sampling_index = pandas.DatetimeIndex(freq="5Min", start=data.index.min(), end=data.index.max())
    event_sampled_df = pandas.DataFrame(index=event_sampling_index)

    ### LTDATA ###
    longterm_data = load_series(find_files(data_dir, "ltdata"))

    # as far as I can tell this doesn't make a difference but it makes me feel better
    longterm_data = longterm_data.resample("1H").mean().interpolate().fillna(method="backfill")

    one_way_latency = get_communication_latency(longterm_data)

    for col in longterm_data.columns:
        add_lag_feature(longterm_data, col, 24, "1d")
        add_lag_feature(longterm_data, col, 4 * 24, "4d")
        add_lag_feature(longterm_data, col, 16 * 24, "16d")
        add_lag_feature(longterm_data, col, 64 * 24, "64d")

    # time-lagged version
    # add_lag_feature(longterm_data, "eclipseduration_min", 2 * 24, "2d", data_type=numpy.int64)
    # add_lag_feature(longterm_data, "eclipseduration_min", 5 * 24, "5d", data_type=numpy.int64)

    ### FTL ###
    ftl_data = load_series(find_files(data_dir, "ftl"), date_cols=["utb_ms", "ute_ms"])

    event_sampled_df["flagcomms"] = get_event_series(event_sampling_index, get_ftl_periods(ftl_data[ftl_data.flagcomms]))
    add_lag_feature(event_sampled_df, "flagcomms", 12, "1h")
    add_lag_feature(event_sampled_df, "flagcomms", 2 * 12, "2h")
    add_lag_feature(event_sampled_df, "flagcomms", 8 * 12, "8h")
    event_sampled_df.drop("flagcomms", axis=1, inplace=True)

    # select columns or take preselected ones
    for ftl_type in ["SLEW", "EARTH", "INERTIAL", "D4PNPO", "MAINTENANCE", "NADIR", "WARMUP", "ACROSS_TRACK"]:
        dest_name = "FTL_" + ftl_type
        event_sampled_df[dest_name] = get_event_series(event_sampled_df.index, get_ftl_periods(ftl_data[ftl_data["type"] == ftl_type]))
        add_lag_feature(event_sampled_df, dest_name, 12, "1h")
        add_lag_feature(event_sampled_df, dest_name, 2 * 12, "2h")
        add_lag_feature(event_sampled_df, dest_name, 8 * 12, "8h")
        add_lag_feature(event_sampled_df, dest_name, 4 * 24 * 12, "4d")
        add_lag_feature(event_sampled_df, dest_name, 16 * 24 * 12, "16d")
        add_lag_feature(event_sampled_df, dest_name, -12 * 2, "next2h")
        # add_lag_feature(event_sampled_df, dest_name, -12 * 8, "next8h")
        event_sampled_df.drop(dest_name, axis=1, inplace=True)

    ### EVTF ###
    event_data = load_series(find_files(data_dir, "evtf"))

    for event_name in ["MAR_UMBRA", "MRB_/_RANGE_06000KM", "MSL_/_RANGE_06000KM"]:
        dest_name = "EVTF_IN_" + event_name
        event_sampled_df[dest_name] = get_event_series(event_sampling_index, get_evtf_ranges(event_data, event_name))
        add_lag_feature(event_sampled_df, dest_name, 12, "1h")
        add_lag_feature(event_sampled_df, dest_name, 12 * 8, "8h")
        # add_lag_feature(event_sampled_df, dest_name, 12 * 32, "32h")
        add_lag_feature(event_sampled_df, dest_name, -12 * 8, "next8h")
        add_lag_feature(event_sampled_df, dest_name, -12 * 32, "next32h")
        event_sampled_df.drop(dest_name, axis=1, inplace=True)

    # event_sampled_df["EVTF_EARTH_LOS"] = get_earth_los(event_data, event_sampled_df.index).rolling(12, min_periods=0).mean()

    event_sampled_df["EVTF_TIME_MRB_AOS_10"] = time_since_last_event(event_data[event_data.description == "MRB_AOS_10"], event_sampled_df.index)
    event_sampled_df["EVTF_TIME_MRB_AOS_00"] = time_since_last_event(event_data[event_data.description == "MRB_AOS_00"], event_sampled_df.index)
    event_sampled_df["EVTF_TIME_MSL_AOS_10"] = time_since_last_event(event_data[event_data.description == "MSL_AOS_10"], event_sampled_df.index)

    altitude_series = get_evtf_altitude(event_data, index=data.index)
    event_data.drop(["description"], axis=1, inplace=True)
    event_data["EVTF_event_counts"] = 1
    event_data = event_data.resample("5Min").count().rolling(12).sum().fillna(method="bfill").reindex(data.index, method="nearest")
    event_data["EVTF_altitude"] = altitude_series
    add_lag_feature(event_data, "EVTF_event_counts", 2, "2h", data_type=numpy.int64)
    add_lag_feature(event_data, "EVTF_event_counts", 5, "5h", data_type=numpy.int64)
    add_lag_feature(event_data, "EVTF_event_counts", 16, "16h", data_type=numpy.int64)
    # add_lag_feature(event_data, "EVTF_event_counts", -5, "next5h", data_type=numpy.int64)

    ### DMOP ###
    dmop_data = load_series(find_files(data_dir, "dmop"))
    adjust_for_latency(dmop_data, one_way_latency)

    dmop_subsystems = get_dmop_subsystem(dmop_data, include_command=False)

    # these subsystems were found partly by trial and error
    for subsys in dmop_subsystems.value_counts().sort_values(ascending=False).index[:15]:
    # for subsys in "PSF ACF MMM TTT SSS HHH OOO MAPO MPER MOCE MOCS PENS PENE TMB VVV SXX".split():
        dest_name = "DMOP_{}_event_count".format(subsys)
        event_sampled_df[dest_name] = event_count(dmop_subsystems[dmop_subsystems == subsys], event_sampled_df.index)
        add_lag_feature(event_sampled_df, dest_name, 12 * 4, "4h")
        add_lag_feature(event_sampled_df, dest_name, 12 * 12, "12h")
        # add_lag_feature(event_sampled_df, dest_name, -12 * 4, "next4h")
        add_lag_feature(event_sampled_df, dest_name, -12 * 12, "next12h")

    # subsystems with the command included just for a few
    dmop_subsystems = get_dmop_subsystem(dmop_data, include_command=True)
    # for subsys in "MMM_F10A0 OOO_F68A0 MMM_F40C0 ACF_E05A PSF_38A1 ACF_M07A ACF_M01A ACF_M06A".split():
    for subsys in dmop_subsystems.value_counts().sort_values(ascending=False).index[:100]:
        dest_name = "DMOP_{}_event_count".format(subsys)
        event_sampled_df[dest_name] = event_count(dmop_subsystems[dmop_subsystems == subsys], event_sampled_df.index)
        add_lag_feature(event_sampled_df, dest_name, 12 * 4, "4h")
        add_lag_feature(event_sampled_df, dest_name, 12 * 16, "16h")
        add_lag_feature(event_sampled_df, dest_name, 12 * 24 * 4, "4d")
        add_lag_feature(event_sampled_df, dest_name, 12 * 24 * 16, "16d")
        add_lag_feature(event_sampled_df, dest_name, 12 * 24 * 64, "64d")

    dmop_data.drop(["subsystem"], axis=1, inplace=True)
    dmop_data["DMOP_event_counts"] = 1
    dmop_data = dmop_data.resample("5Min").count().rolling(12).sum().fillna(method="bfill").reindex(data.index, method="nearest")
    add_lag_feature(dmop_data, "DMOP_event_counts", 2, "2h", data_type=numpy.int64)
    add_lag_feature(dmop_data, "DMOP_event_counts", 5, "5h", data_type=numpy.int64)
    add_lag_feature(dmop_data, "DMOP_event_counts", 12, "12h", data_type=numpy.int64)
    # add_lag_feature(dmop_data, "DMOP_event_counts", -5, "next5h", data_type=numpy.int64)


    ### SAAF ###
    saaf_data = load_series(find_files(data_dir, "saaf"))

    # try a totally different style
    # saaf_data = saaf_data.resample("15Min").mean().interpolate().rolling(4).mean().fillna(method="bfill").reindex(data.index, method="nearest")
    saaf_data = saaf_data.resample("2Min").mean().interpolate()
    saaf_periods = 30

    saaf_quartiles = []
    for col in ["sx", "sy", "sz", "sa"]:
        quartile_indicator_df = pandas.get_dummies(pandas.qcut(saaf_data[col], 10), col + "_")
        quartile_hist_df = quartile_indicator_df.rolling(saaf_periods, min_periods=1).mean()
        saaf_quartiles.append(quartile_hist_df)

    # feature_weights = [(u'sz__(85.86, 90.0625]', 0.029773711557758643),
    #                    (u'sa__(1.53, 5.192]', 0.018961685024380826),
    #                    (u'sy__(89.77, 89.94]', 0.017065361875026698),
    #                    (u'sx__(2.355, 5.298]', 0.016407419574092953),
    #                    (u'sx__(32.557, 44.945]', 0.015424066115462104),
    #                    (u'sz__(101.35, 107.00167]', 0.012587397977008816),
    #                    (u'sz__(112.47, 117.565]', 0.010126162391495826),
    #                    (u'sz__(107.00167, 112.47]', 0.0099830744754197034),
    #                    (u'sz__(117.565, 121.039]', 0.0081198807115996485),
    #                    (u'sa__(5.192, 18.28]', 0.0079614117402690421),
    #                    (u'sy__(89.94, 90]', 0.0064161463399279106),
    #                    (u'sz__(90.0625, 95.455]', 0.0060623602567580299),
    #                    (u'sa__(0.739, 1.53]', 0.0050941311789206674),
    #                    (u'sa__(0.198, 0.31]', 0.00064943741967410915)]
    #
    # for feature, weight in feature_weights:
    #     if feature.startswith("s") and "__" in feature:
    #         base, lower, upper = parse_cut_feature(feature)
    #
    #         indicators = (saaf_data[base] > lower) & (saaf_data[base] <= upper)
    #         rolling_count = indicators.rolling(saaf_periods, min_periods=1).mean()
    #         rolling_count.rename(feature, inplace=True)
    #         saaf_quartiles.append(rolling_count)

    saaf_quartile_df = pandas.concat(saaf_quartiles, axis=1)

    for col in saaf_quartile_df.columns:
        add_lag_feature(saaf_quartile_df, col, saaf_periods * 4, "4h")
        add_lag_feature(saaf_quartile_df, col, saaf_periods * 12, "12h")
        add_lag_feature(saaf_quartile_df, col, saaf_periods * 48, "48h")

    saaf_quartile_df = saaf_quartile_df.reindex(data.index, method="nearest")

    # convert to simple rolling mean
    saaf_data = saaf_data.rolling(saaf_periods).mean().fillna(method="bfill")

    # SAAF rolling stddev, took top 2 from ElasticNet
    for num_days in [1, 8]:
        saaf_data["SAAF_stddev_{}d".format(num_days)] = saaf_data[["sx", "sy", "sz", "sa"]].rolling(num_days * 24 * saaf_periods).std().fillna(method="bfill").sum(axis=1)
    saaf_data = saaf_data.reindex(data.index, method="nearest").fillna(method="bfill")

    longterm_data = longterm_data.reindex(data.index, method="nearest")

    data = pandas.concat([data, saaf_data, longterm_data, dmop_data, event_data, event_sampled_df.reindex(data.index, method="nearest"), saaf_quartile_df], axis=1)
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

        add_transformation_feature(data, "sa", "log", drop=True)
        add_transformation_feature(data, "sy", "log", drop=True)

        # # various crazy rolling features
        add_lag_feature(data, "EVTF_IN_MAR_UMBRA_rolling_1h", 50, "50")
        add_lag_feature(data, "EVTF_IN_MRB_/_RANGE_06000KM_rolling_1h", 1600, "1600")
        add_lag_feature(data, "EVTF_event_counts_rolling_5h", 50, "50")
        # add_lag_feature(data, "FTL_ACROSS_TRACK_rolling_1h", 200, "200")
        add_lag_feature(data, "FTL_NADIR_rolling_1h", 400, "400")

    data.drop("earthmars_km", axis=1, inplace=True)

    logger.info("DataFrame shape %s", data.shape)
    return data

def cross_validated_select(dataset, splits, feature_scoring_function, std_dev_weight=-.05):
    scores = []
    for train, _ in splits:
        scores.append(feature_scoring_function(dataset.inputs[train], dataset.outputs[train]))

    score_matrix = numpy.vstack(scores)
    return score_matrix.mean(axis=0) + std_dev_weight * score_matrix.std(axis=0)


def multivariate_select(dataset, feature_scoring_function, weight_outputs=False):
    output_weights = dataset.outputs.mean(axis=0) + dataset.outputs.std(axis=0)

    scores = []
    for output_index in range(dataset.outputs.shape[1]):
        output = dataset.outputs[:, output_index]

        scores.append(feature_scoring_function(dataset.inputs, output))

    score_matrix = numpy.vstack(scores) # M outputs x N features

    if weight_outputs:
        # should be 1 x N
        return output_weights.dot(score_matrix)
    else:
        return score_matrix.mean(axis=0)


def ensemble_feature_scores(*scores):
    return numpy.vstack(scores).prod(axis=0)


def test_models(dataset, name, with_nn=False):
    print("Selecting features with {}, {} features".format(name, dataset.inputs.shape[1]))
    cross_validate(dataset, with_scaler(sklearn.linear_model.ElasticNet(0.001), "en"))

    if with_nn:
        cross_validate(dataset, with_scaler(make_nn()[0], "nn"))


def make_select_f(num_features, ewma=False):
    def score_features_f(X, y):
        selector = sklearn.feature_selection.SelectKBest(score_func=sklearn.feature_selection.f_regression, k=num_features)

        if ewma:
            y = centered_ewma(pandas.Series(y), 30).values
        selector.fit(X, y)
        return selector.scores_

    return score_features_f


def test_simple(dataset, num_features):
    selector = sklearn.feature_selection.SelectKBest(score_func=sklearn.feature_selection.f_regression, k=num_features)
    selector.fit(dataset.inputs, dataset.outputs.sum(axis=1))
    reduced_dataset = dataset.select_features(num_features, selector.scores_)
    test_models(reduced_dataset, "f_regression(sum)", with_nn=False)

    # try with ewma
    reduced_dataset = dataset.select_features(num_features, make_select_f(num_features, True)(dataset.inputs, dataset.outputs.sum(axis=1)))
    test_models(reduced_dataset, "f_regression_ewma(sum)", with_nn=False)


def test_simple_multivariate(dataset, num_features):
    scores = multivariate_select(dataset, make_select_f(num_features))
    reduced_dataset = dataset.select_features(num_features, scores)
    test_models(reduced_dataset, "f_regression(multivariate)", with_nn=False)

    scores = multivariate_select(dataset, make_select_f(num_features), weight_outputs=True)
    print(scores.shape)
    reduced_dataset = dataset.select_features(num_features, scores)
    test_models(reduced_dataset, "f_regression(weighted multivariate)", with_nn=False)


def test_select_from_en(dataset, num_features):
    model = with_scaler(sklearn.linear_model.ElasticNet(0.001), "en")
    model.fit(dataset.inputs, dataset.outputs.sum(axis=1))

    reduced_dataset = dataset.select_features(num_features, abs(model.named_steps["en"].coef_), verbose=1)
    test_models(reduced_dataset, "ElasticNet(sum)")


def score_features_elasticnet(X, Y):
    model = with_scaler(sklearn.linear_model.ElasticNet(0.001), "en")
    model.fit(X, Y.sum(axis=1))
    return abs(model.named_steps["en"].coef_)


def score_features_loo(X, Y, splits, scorer, std_dev_weight=-.05, model_override=None, prob_test=1.):
    import random

    if not model_override:
        model = with_scaler(sklearn.linear_model.Ridge(), "ridge")
    else:
        model = model_override
    scores = [0 for _ in range(X.shape[1])]
    baseline = sklearn.cross_validation.cross_val_score(model, X, Y, scoring=scorer, cv=splits).mean()
    for i in range(X.shape[1]):
        if prob_test < 1 and random.random() < prob_test:
            scores[i] = None
            continue

        included = numpy.asarray([j for j in range(X.shape[1]) if j != i])
        cv_scores = sklearn.cross_validation.cross_val_score(model, X[:, included], Y, scoring=scorer, cv=splits)
        scores[i] = baseline - cv_scores.mean() + std_dev_weight * cv_scores.std()

    # fill any NA
    mean_score = numpy.mean([score for score in scores if score is not None])
    scores = [score if score is not None else mean_score for score in scores]

    return numpy.asarray(scores)


def score_features_loi(X, Y, splits, scorer, std_dev_weight=-.05):
    model = with_scaler(sklearn.linear_model.Ridge(), "ridge")
    scores = [0 for _ in range(X.shape[1])]
    for i in range(X.shape[1]):
        cv_scores = sklearn.cross_validation.cross_val_score(model, X[:, [i]], Y, scoring=scorer, cv=splits)
        scores[i] = cv_scores.mean() + std_dev_weight * cv_scores.std()

    return numpy.asarray(scores)


def test_select_from_en_cv(dataset, num_features, splits):
    scores = cross_validated_select(dataset, splits, score_features_elasticnet)
    reduced_dataset = dataset.select_features(num_features, scores, verbose=1)
    test_models(reduced_dataset, "CV(ElasticNet(sum))")


def test_select_from_cv2(dataset, num_features, splits, std_dev_weight=-.05):
    model = with_scaler(sklearn.linear_model.ElasticNet(0.001), "en")

    scores = []
    for train, test in splits:
        model.fit(dataset.inputs[train], dataset.outputs[train])
        scores.append(abs(model.named_steps["en"].coef_).mean(axis=0) / -rms_error(model, dataset.inputs[test], dataset.outputs[test]))

    score_matrix = numpy.vstack(scores)
    merged_scores = score_matrix.mean(axis=0) + std_dev_weight * score_matrix.std(axis=0)

    reduced_dataset = dataset.select_features(num_features, merged_scores, verbose=1)
    test_models(reduced_dataset, "CV+Test(ElasticNet(sum))")


def test_select_from_rf(dataset, num_features):
    model = sklearn.ensemble.RandomForestRegressor(40, max_depth=20)
    model.fit(dataset.inputs, dataset.outputs.sum(axis=1))

    reduced_dataset = dataset.select_features(num_features, model.feature_importances_, verbose=1)
    test_models(reduced_dataset, "RandomForest(sum)")

def select_nonzero(dataset):
    model = with_scaler(sklearn.linear_model.ElasticNet(0.001), "en")
    model.fit(dataset.inputs, dataset.outputs)
    scores = abs(model.named_steps["en"].coef_).mean(axis=0)
    return dataset.select_nonzero_features(scores, verbose=1)


def test_rfecv_en(dataset, num_features, tuning_splits, prefilter=True):
    model = sklearn.linear_model.ElasticNet(0.001)

    if prefilter:
        dataset = select_nonzero(dataset)

    selector = sklearn.feature_selection.RFECV(model, cv=tuning_splits, step=5, scoring=rms_error)

    X = helpers.sk.ClippedRobustScaler().fit_transform(dataset.inputs)
    selector.fit(X, dataset.outputs.mean(axis=1))

    reduced_dataset = dataset.select_features(num_features, selector.ranking_, higher_is_better=False, verbose=1)
    test_models(reduced_dataset, "RFECV(ElasticNet(sum))")

    model.fit(X, dataset.outputs.sum(axis=1))
    ensembled_weights = abs(model.coef_) / selector.ranking_
    reduced_dataset = dataset.select_features(num_features, ensembled_weights, verbose=1)
    test_models(reduced_dataset, "Ens of RFECV+EN(ElasticNet(sum))")

    ensembled_weights = abs(model.coef_) * .75 ** selector.ranking_
    reduced_dataset = dataset.select_features(num_features, ensembled_weights, verbose=1)
    test_models(reduced_dataset, "Ens of RFECV+EN(ElasticNet(sum)) V2")

    # ensemble of leave one out, leave one in, coef but no zero clip, RFECV
    loi_scores = score_features_loi(dataset.inputs, dataset.outputs, tuning_splits, rms_error)
    loo_scores = score_features_loo(dataset.inputs, dataset.outputs, tuning_splits, rms_error)
    ensembled_weights = .5 ** loi_scores * .5 ** loo_scores * (abs(model.coef_) + .1) / selector.ranking_
    reduced_dataset = dataset.select_features(num_features, ensembled_weights, verbose=1)
    test_models(reduced_dataset, "Ens of RFECV+EN+LOI+LOO(ElasticNet(sum)) V2")


def test_simple_ensemble(dataset, num_features, splits):
    en_cv_scores = cross_validated_select(dataset, splits, score_features_elasticnet)

    # ensemble of leave one out, leave one in, coef but no zero clip, RFECV
    loi_scores = score_features_loi(dataset.inputs, dataset.outputs, splits, rms_error)

    scores = .5 ** loi_scores * (en_cv_scores + .1)
    reduced_dataset = dataset.select_features(num_features, scores, verbose=1)
    test_models(reduced_dataset, "Ensemble of ENCV+LOI")


def test_subspace_selection(dataset, num_features, splits, prefilter=True):
    if prefilter:
        dataset = select_nonzero(dataset)

    orig_num_features = dataset.inputs.shape[1]

    while dataset.inputs.shape[1] > num_features * 2:
        model = helpers.sk.MultivariateBaggingRegressor(with_scaler(sklearn.linear_model.Ridge(), "rr"), max_features=16, n_estimators=orig_num_features)
        feature_scores = model.evaluate_features_cv(dataset.inputs, dataset.outputs, splits)
        dataset = dataset.select_features(0.75, feature_scores, higher_is_better=False)

    model = helpers.sk.MultivariateBaggingRegressor(with_scaler(sklearn.linear_model.Ridge(), "rr"), max_features=num_features, n_estimators=orig_num_features)
    feature_scores = model.evaluate_features_cv(dataset.inputs, dataset.outputs, splits)
    dataset = dataset.select_features(num_features, feature_scores, higher_is_better=False)

    test_models(dataset, "subspace elimination")


def test_loo_loi(dataset, num_features, splits):
    test_models(dataset.select_features(num_features, score_features_loi(dataset.inputs, dataset.outputs, splits, rms_error), verbose=1), "leave one in")

    test_models(dataset.select_features(num_features, score_features_loo(dataset.inputs, dataset.outputs, splits, rms_error), verbose=1), "leave one out")

def rfe_eval(estimator, dataset, num_features, splits, step=2):
    current_data = dataset
    while current_data.inputs.shape[1] - step >= num_features:
        # weed out S features with LOO
        scores = score_features_loo(current_data.inputs, current_data.outputs, splits, rms_error, model_override=estimator, prob_test=0.5)
        current_data = current_data.select_features(current_data.inputs.shape[1] - step, scores, verbose=1)

    # eliminate the rest
    leftover = current_data.inputs.shape[1] - num_features
    if leftover:
        scores = score_features_loo(current_data.inputs, current_data.outputs, splits, rms_error, model_override=estimator, prob_test=0.5)
        current_data = current_data.select_features(leftover, scores, verbose=1)

    selected_features = set(current_data.feature_names)
    feature_indicators = numpy.asarray([f in selected_features for f in dataset.feature_names], dtype=numpy.int8)
    return feature_indicators


def test_rfe_hybrid(dataset, num_features, splits, preselect=True):
    # select down to num_features * 2 using a reasonable but naive method
    if preselect:
        scores = cross_validated_select(dataset, splits, score_features_elasticnet)
        dataset = dataset.select_features(num_features * 2, scores, verbose=1)

    test_models(dataset, "EN-CV(sum) to features * 2")

    # select the rest of the way down with RFE LOO on MLP
    rfe_eval(None, dataset, num_features, splits, step=3)

    # evaluate
    test_models(dataset.select_features(num_features, score_features_loi(dataset.inputs, dataset.outputs, splits, rms_error), verbose=1), "LOI-RFE/LOO hybrid")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    dataset = load_split_data(args, data_loader=load_data)

    print("Baselines")
    cross_validate(dataset, sklearn.dummy.DummyRegressor())
    test_models(dataset, "baseline", with_nn=False)

    tuning_splits = sklearn.cross_validation.KFold(dataset.inputs.shape[0], 3, False)


    test_loo_loi(dataset, args.num_features, tuning_splits)
    test_rfe_hybrid(dataset, args.num_features, tuning_splits, preselect=False)

    # test_simple_multivariate(dataset, args.num_features)
    # test_select_from_cv2(dataset, args.num_features, tuning_splits)
    # test_simple(dataset, args.num_features)

    test_select_from_en(dataset, args.num_features)
    test_select_from_en_cv(dataset, args.num_features, tuning_splits)
    test_simple_ensemble(dataset, args.num_features, tuning_splits)
    test_rfecv_en(dataset, args.num_features, tuning_splits)
    test_subspace_selection(dataset, args.num_features, tuning_splits)

if __name__ == "__main__":
    sys.exit(main())
