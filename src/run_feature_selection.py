from __future__ import print_function
from __future__ import unicode_literals

import argparse
import collections
import logging
import re
import sys
from operator import itemgetter

import numpy
import sklearn.cross_validation
import sklearn.dummy
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.linear_model

import helpers.general
import helpers.sk
from helpers.feature_selection import score_features_random_forest
from helpers.sk import rms_error
from loaders import load_split_data, get_loader, add_loader_arguments
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


def test_models(dataset, name, with_nn=True, with_rnn=False):
    print("Evaluating {}, {} features".format(name, dataset.inputs.shape[1]))
    cross_validate(dataset, with_scaler(sklearn.linear_model.ElasticNet(0.001), "en"))

    if with_nn:
        cross_validate(dataset, with_scaler(make_nn()[0], "nn"))

    if with_rnn:
        cross_validate(dataset, with_scaler(with_non_negative(make_rnn()[0]), "rnn"))


def test_select_from_en(dataset, num_features):
    model = with_scaler(sklearn.linear_model.ElasticNet(0.001), "en")
    model.fit(dataset.inputs, dataset.outputs.sum(axis=1))

    reduced_dataset = dataset.select_features(num_features, abs(model.named_steps["en"].coef_), verbose=1)
    test_models(reduced_dataset, "ElasticNet(sum)")


def score_features_elastic_net(X, Y):
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


def score_features_loi(X, Y, splits, scorer, std_dev_weight=-.05, model="linear", noise=None):
    if model == "linear":
        model = with_scaler(sklearn.linear_model.Ridge(), "ridge")
    elif model == "rnn":
        model = with_scaler(with_non_negative(make_rnn()[0]), "rnn")
    else:
        raise ValueError("Unsupported model for score_features_loi: {}".format(model))

    if noise:
        X = helpers.general.add_temporal_noise(X, noise)

    scores = [0 for _ in range(X.shape[1])]
    for i in range(X.shape[1]):
        cv_scores = sklearn.cross_validation.cross_val_score(model, X[:, [i]], Y, scoring=scorer, cv=splits)
        scores[i] = cv_scores.mean() + std_dev_weight * cv_scores.std()

    return numpy.asarray(scores)


def deform_feature(X, i, adjustment=0.1):
    X_def = X.copy()
    X_def[:, i] *= 1 + adjustment
    return X_def


def score_features_deform(X, Y, splits, std_dev_weight=-.05, model_type="linear"):
    if model_type == "linear":
        model = with_scaler(sklearn.linear_model.ElasticNet(0.001), "en")
    elif model_type == "nn":
        model = with_scaler(with_non_negative(make_nn()[0]), "nn")
    else:
        raise ValueError("Unsupported model for score_features_loi: {}".format(model_type))

    all_degradation_scores = []
    for train, test in splits:
        model.fit(X[train], Y[train])

        baseline_score = rms_error(model, X[test], Y[test])
        deform_scores_increased = numpy.asarray([rms_error(model, deform_feature(X[test], i, adjustment=0.1), Y[test]) for i in range(X.shape[1])])
        deform_scores_decreased = numpy.asarray([rms_error(model, deform_feature(X[test], i, adjustment=-0.1), Y[test]) for i in range(X.shape[1])])

        all_degradation_scores.append(baseline_score - deform_scores_increased)
        all_degradation_scores.append(baseline_score - deform_scores_decreased)

    all_degradation_scores = numpy.asarray(all_degradation_scores)
    feature_scores = all_degradation_scores.mean(axis=0) + std_dev_weight * all_degradation_scores.std(axis=0)

    return feature_scores


def test_select_from_en_cv(dataset, num_features, splits):
    scores = cross_validated_select(dataset, splits, score_features_elastic_net)
    reduced_dataset = dataset.select_features(num_features, scores, verbose=1)
    test_models(reduced_dataset, "CV(ElasticNet(sum))")

    diversified_scores = diversify(dataset.feature_names, scores)
    reduced_dataset = dataset.select_features(num_features, diversified_scores, verbose=1)
    test_models(reduced_dataset, "DIVERSIFY! CV(ElasticNet(sum))")


def test_simple_ensemble(dataset, num_features, splits, loi_model="linear"):
    en_cv_scores = cross_validated_select(dataset, splits, score_features_elastic_net)

    # ensemble of leave one out, leave one in, coef but no zero clip, RFECV
    loi_scores = score_features_loi(dataset.inputs, dataset.outputs, splits, rms_error, model=loi_model)

    scores = .5 ** loi_scores * (en_cv_scores + .1)
    reduced_dataset = dataset.select_features(num_features, scores, verbose=1)
    test_models(reduced_dataset, "Ensemble of ENCV+LOI({})".format(loi_model))


def test_cv_ensemble(dataset, num_features, splits):
    scores1 = cross_validated_select(dataset, splits, score_features_elastic_net)
    scores2 = cross_validated_select(dataset, dataset.splits, score_features_elastic_net)

    scores = (scores1 + .1) * (scores2 + .1)
    test_models(dataset.select_features(num_features, scores, verbose=1), "Ensemble of ENCV on both CV splits")

    diversified_scores = diversify(dataset.feature_names, scores)
    test_models(dataset.select_features(num_features, diversified_scores, verbose=1), "Diversified Ensemble of ENCV on both CV splits")


def test_cv_noise_ensemble(dataset, num_features, noise_list=[0.03], noise_iter=1, score_function=score_features_elastic_net, stddev_weight=0.05, score_cv=False):
    noise_list = noise_list * noise_iter
    scores = []

    for noise in noise_list:
        noised_data = with_noise(dataset, noise, temporal=True)
        for cv in dataset.split_map.values():
            if not score_cv:
                scores.append(cross_validated_select(noised_data, cv, score_function))
            else:
                scores.append(score_function(dataset.inputs, dataset.outputs, cv, rms_error))

    joined_scores = numpy.vstack(scores)
    scores = joined_scores.mean(axis=0) - stddev_weight * joined_scores.std(axis=0)
    test_models(dataset.select_features(num_features, scores, verbose=1), "Noise*CV ensemble {}".format(score_function.__name__))

    diversified_scores = diversify(dataset.feature_names, scores)
    test_models(dataset.select_features(num_features, diversified_scores, verbose=1), "Diversified Noise*CV ensemble {}".format(score_function.__name__))


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

def with_noise(dataset, noise, temporal=False):
    """Make a variant of this dataset with noise in the inputs"""
    if temporal:
        noised_inputs = helpers.general.add_temporal_noise(dataset.inputs, noise)
    else:
        noise_multipliers = noise * (numpy.random.rand(*dataset.inputs.shape) - 0.5) + 1
        noised_inputs = dataset.inputs * noise_multipliers

    return helpers.general.DataSet(noised_inputs, dataset.outputs, dataset.splits, dataset.feature_names, dataset.target_names, dataset.output_index)


def test_noise_insensitivity(dataset, num_features, splits, noise=0.01, num_noise=3, nonparametric=False):
    """Test the features on a model with noise variations"""
    scores = []
    for i in range(num_noise):
        scores.append(cross_validated_select(with_noise(dataset, noise), splits, score_features_elastic_net))

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
    test_models(dataset, "baseline", with_nn=True, with_rnn=False)

    tuning_splits = dataset.split_map["alexcv"]

    # scoring with deform
    test_models(dataset.select_features(args.num_features, score_features_deform(dataset.inputs, dataset.outputs, tuning_splits), verbose=1), "deformed EN")
    test_models(dataset.select_features(args.num_features, score_features_deform(dataset.inputs, dataset.outputs, tuning_splits, model_type="nn"), verbose=1), "deformed NN")

    # top priority = LOI
    test_models(dataset.select_features(args.num_features, score_features_loi(dataset.inputs, dataset.outputs, tuning_splits, rms_error), verbose=1), "leave one in")
    test_models(dataset.select_features(args.num_features, score_features_loi(dataset.inputs, dataset.outputs, tuning_splits, rms_error, noise=0.1), verbose=1), "leave one in, 10% temporal noise")
    test_models(dataset.select_features(args.num_features, score_features_loi(dataset.inputs, dataset.outputs, tuning_splits, rms_error, noise=0.5), verbose=1), "leave one in, 50% temporal noise")

    # various ensembles
    test_cv_noise_ensemble(dataset, args.num_features, noise_list=[0.1], noise_iter=5)
    test_cv_noise_ensemble(dataset, args.num_features, noise_list=[0.05, 0.1, 0.25], score_function=score_features_loi, score_cv=True)
    test_cv_noise_ensemble(dataset, args.num_features, noise_list=[0.05, 0.1, 0.25], score_function=score_features_random_forest)


if __name__ == "__main__":
    sys.exit(main())
