from __future__ import print_function
from __future__ import unicode_literals

import argparse
import collections
import logging
import re
import sys
from operator import itemgetter

import numpy
import pandas
import scipy.stats
import sklearn.cross_validation
import sklearn.dummy
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.linear_model

import helpers.general
import helpers.sk
from helpers.sk import rms_error
from loaders import centered_ewma, load_split_data, get_loader, add_loader_arguments
from train_test import make_nn, make_rnn, cross_validate, with_scaler, with_non_negative


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_loader_arguments(parser)
    parser.add_argument("num_features", default=40, type=int, help="Number of features to select")
    parser.add_argument("training_dir", help="Dir with the training CSV files or joined CSV file with the complete feature matrix")
    return parser.parse_args()


def split_feature_name(name):
    feature_pattern = re.compile(r"(.*)_((?:next)?\d+[dhm])")
    modifier_pattern = re.compile(r"_(tan|log|rolling)(?=_|$)")

    name = modifier_pattern.sub("", name)

    match = feature_pattern.match(name)
    if match:
        return match.group(1), match.group(2)

    return name, None


def diversify(feature_names, feature_scores, decay=0.8):
    """Encourage diversity by trying not to select features from the same groups quite so much"""
    current_index = collections.defaultdict(int)
    modifiers = dict()

    for name, score in sorted(zip(feature_names, feature_scores), key=itemgetter(1), reverse=True):
        base_feature, time_component = split_feature_name(name)
        if time_component and time_component.startswith("next"):
            base_feature += "_next"

        modifiers[name] = decay ** current_index[base_feature]
        current_index[base_feature] += 1

    return [score * modifiers[name] for name, score in zip(feature_names, feature_scores)]


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


def test_models(dataset, name, with_nn=True, with_rnn=True):
    print("Evaluating {}, {} features".format(name, dataset.inputs.shape[1]))
    cross_validate(dataset, with_scaler(sklearn.linear_model.ElasticNet(0.001), "en"))

    if with_nn:
        cross_validate(dataset, with_scaler(make_nn()[0], "nn"))

    if with_rnn:
        cross_validate(dataset, with_scaler(make_rnn(non_negative=True)[0], "rnn"))
        cross_validate(dataset, with_scaler(with_non_negative(make_rnn()[0]), "rnn"))


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


def score_features_ridge(X, Y):
    model = with_scaler(sklearn.linear_model.Ridge(0.01), "rr")
    model.fit(X, Y.sum(axis=1))
    return abs(model.named_steps["rr"].coef_)


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


def score_features_loi(X, Y, splits, scorer, std_dev_weight=-.05, model="linear"):
    if model == "linear":
        model = with_scaler(sklearn.linear_model.Ridge(), "ridge")
    elif model == "rnn":
        model = with_scaler(with_non_negative(make_rnn()[0]), "rnn")
    else:
        raise ValueError("Unsupported model for score_features_loi: {}".format(model))

    scores = [0 for _ in range(X.shape[1])]
    for i in range(X.shape[1]):
        cv_scores = sklearn.cross_validation.cross_val_score(model, X[:, [i]], Y, scoring=scorer, cv=splits)
        scores[i] = cv_scores.mean() + std_dev_weight * cv_scores.std()

    return numpy.asarray(scores)

def inverse_rank_order(weights):
    return 0.9 ** scipy.stats.rankdata(weights)

def test_select_from_en_cv(dataset, num_features, splits):
    scores = cross_validated_select(dataset, splits, score_features_elasticnet)
    reduced_dataset = dataset.select_features(num_features, scores, verbose=1)
    test_models(reduced_dataset, "CV(ElasticNet(sum))")

    diversified_scores = diversify(dataset.feature_names, scores)
    reduced_dataset = dataset.select_features(num_features, diversified_scores, verbose=1)
    test_models(reduced_dataset, "DIVERSIFY! CV(ElasticNet(sum))")


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


def test_simple_ensemble(dataset, num_features, splits, loi_model="linear"):
    en_cv_scores = cross_validated_select(dataset, splits, score_features_elasticnet)

    # ensemble of leave one out, leave one in, coef but no zero clip, RFECV
    loi_scores = score_features_loi(dataset.inputs, dataset.outputs, splits, rms_error, model=loi_model)

    scores = .5 ** loi_scores * (en_cv_scores + .1)
    reduced_dataset = dataset.select_features(num_features, scores, verbose=1)
    test_models(reduced_dataset, "Ensemble of ENCV+LOI({})".format(loi_model))

def test_cv_ensemble(dataset, num_features, splits):
    scores1 = cross_validated_select(dataset, splits, score_features_elasticnet)
    scores2 = cross_validated_select(dataset, dataset.splits, score_features_elasticnet)

    scores = (scores1 + .1) * (scores2 + .1)
    test_models(dataset.select_features(num_features, scores, verbose=1), "Ensemble of ENCV on both CV splits")

    diversified_scores = diversify(dataset.feature_names, scores)
    test_models(dataset.select_features(num_features, diversified_scores, verbose=1), "Diversified Ensemble of ENCV on both CV splits")


def test_mega_ensemble(dataset, num_features, splits, noise=0.03, noise_iter=5):
    scorers = [score_features_elasticnet]

    scores = []
    for _ in range(noise_iter):
        noised_data = with_noise(dataset, noise)
        for cv in dataset.split_map.values():
            for scorer in scorers:
                scores.append(cross_validated_select(noised_data, cv, scorer))

    joined_scores = numpy.vstack(scores)
    scores = joined_scores.mean(axis=0) - 0.05 * joined_scores.std(axis=0)
    test_models(dataset.select_features(num_features, scores, verbose=1), "Noise*CV*model ensemble")

    diversified_scores = diversify(dataset.feature_names, scores)
    test_models(dataset.select_features(num_features, diversified_scores, verbose=1), "Diversified Noise*CV*model ensemble")


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


def test_subspace_simple(dataset, num_features, splits, num_iter=500):
    best_weights = None
    best_score = None

    all_scores = []

    for i in range(num_iter):
        weights = numpy.random.rand(dataset.inputs.shape[1], 1).flatten()
        reduced = dataset.select_features(num_features, weights)
        score = sklearn.cross_validation.cross_val_score(sklearn.linear_model.ElasticNet(0.0001), reduced.inputs, dataset.outputs, scoring=rms_error, cv=splits, n_jobs=1).mean()

        if not best_score or score > best_score:
            best_score = score
            best_weights = weights

        pairs = sorted(enumerate(weights), key=itemgetter(1), reverse=True)[:num_features]
        all_scores.append((score, map(itemgetter(0), pairs)))

    reduced_dataset = dataset.select_features(num_features, best_weights)
    test_models(reduced_dataset, "subspace elimination (simplified)")

    diversified_scores = diversify(dataset.feature_names, best_weights)
    reduced_dataset = dataset.select_features(num_features, diversified_scores)
    test_models(reduced_dataset, "Diversified subspace elimination (simplified)")

    weights = count_important_features(all_scores, dataset)
    reduced_dataset = dataset.select_features(num_features, weights)
    test_models(reduced_dataset, "subspace elimination, simple merger of top feature sets")


def count_important_features(all_scores, dataset, fraction_models=0.05):
    all_scores = sorted(all_scores, key=itemgetter(0), reverse=True)
    feature_counts = collections.Counter()
    for _, feature_indexes in all_scores[:int(len(all_scores) * fraction_models)]:
        for i in feature_indexes:
            feature_counts[i] += 1

    return numpy.asarray([feature_counts[i] for i in range(dataset.inputs.shape[1])])


def test_subspace_mlp(dataset, num_features, splits, num_iter=100):
    best_weights = None
    best_score = None

    model, _ = make_nn()

    all_scores = []

    for i in range(num_iter):
        weights = numpy.random.rand(dataset.inputs.shape[1], 1).flatten()
        reduced = dataset.select_features(num_features, weights)
        score = sklearn.cross_validation.cross_val_score(model, reduced.inputs, dataset.outputs, scoring=rms_error, cv=splits, n_jobs=1).mean()

        if not best_score or score > best_score:
            best_score = score
            best_weights = weights

        pairs = sorted(enumerate(weights), key=itemgetter(1), reverse=True)[:num_features]
        all_scores.append((score, map(itemgetter(0), pairs)))

    dataset = dataset.select_features(num_features, best_weights)
    test_models(dataset, "subspace testing with nn")

    diversified_scores = diversify(dataset.feature_names, best_weights)
    reduced_dataset = dataset.select_features(num_features, diversified_scores)
    test_models(reduced_dataset, "Diversified subspace testing with nn (simplified)")

    # second version that uses the top 5%
    weights = count_important_features(all_scores, dataset)
    reduced_dataset = dataset.select_features(num_features, weights)
    test_models(reduced_dataset, "subspace testing with nn, simple merger of top feature sets")




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
    scores = rfe_eval(None, dataset, num_features, splits, step=3)

    # evaluate
    test_models(dataset.select_features(num_features, scores, verbose=1), "RFE/LOO hybrid")


def with_noise(dataset, noise):
    noise_multipliers = noise * (numpy.random.rand(*dataset.inputs.shape) - 0.5) + 1
    return helpers.general.DataSet(dataset.inputs * noise_multipliers, dataset.outputs, dataset.splits, dataset.feature_names, dataset.target_names, dataset.output_index)


def test_noise_insensitivity(dataset, num_features, splits, noise=0.01, num_noise=3, nonparametric=False):
    """Test the features on a model with noise variations"""
    scores = []
    for i in range(num_noise):
        scores.append(cross_validated_select(with_noise(dataset, noise), splits, score_features_elasticnet))

    if nonparametric:
        scores = [inverse_rank_order(s) for s in scores]

    score_matrix = numpy.vstack(scores)
    scores = score_matrix.mean(axis=0)
    test_models(dataset.select_features(num_features, scores, verbose=1), "NoisyCV_{}@{}{}(ElasticNet(sum))".format(num_noise, noise, "nonpara" if nonparametric else ""))


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    dataset = load_split_data(args, data_loader=get_loader(args))
    # dataset.split_map = None

    print("Baselines")
    cross_validate(dataset, sklearn.dummy.DummyRegressor())
    test_models(dataset, "baseline", with_nn=False, with_rnn=False)

    tuning_splits = dataset.split_map["alexcv"]

    # data size * total features * cv (1 pass)
    # test_select_from_en_cv(dataset, args.num_features, tuning_splits)

    # data size * total features * cv * num noise
    # test_noise_insensitivity(dataset, args.num_features, tuning_splits)
    test_noise_insensitivity(dataset, args.num_features, tuning_splits, noise=0.03)
    test_mega_ensemble(dataset, args.num_features, tuning_splits, noise=0.03)

    # data size * total features * cv * 2
    # test_cv_ensemble(dataset, args.num_features, tuning_splits)

    # data size * total features * 1
    # test_select_from_en(dataset, args.num_features)

    # reduced features * data size * cv * iter
    test_subspace_simple(dataset, args.num_features, tuning_splits, num_iter=150)
    # test_subspace_mlp(dataset, args.num_features, tuning_splits, num_iter=150)

    # loi = 1 * total features * data size * cv
    # loo = total features * total - 1 * data size * cv
    test_loo_loi(dataset, args.num_features, tuning_splits)

    # loi + en-cv
    test_simple_ensemble(dataset, args.num_features, tuning_splits)
    # test_simple_ensemble(dataset, args.num_features, tuning_splits, loi_model="rnn")

    # super slow methods
    # test_rfecv_en(dataset, args.num_features, tuning_splits)
    # test_subspace_selection(dataset, args.num_features, tuning_splits)
    # test_rfe_hybrid(dataset, args.num_features, tuning_splits, preselect=False)

if __name__ == "__main__":
    sys.exit(main())
