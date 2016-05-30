from __future__ import print_function
from __future__ import unicode_literals

import io

from train_test import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", default=False, action="store_true", help="Run verifications on the input data for outliers and such")
    parser.add_argument("--resample", default="1H", help="Time interval to resample the training data")
    parser.add_argument("--extra-analysis", default=False, action="store_true", help="Extra analysis on the data")
    parser.add_argument("num_features", default=40, type=int, help="Number of features to select")
    parser.add_argument("training_dir", help="Dir with the training CSV files or joined CSV file with the complete feature matrix")
    return parser.parse_args()

def test_models(dataset, name):
    print("Testing models with {}, {} features".format(name, dataset.inputs.shape[1]))
    cross_validate(dataset, with_scaler(sklearn.linear_model.ElasticNet(0.001), "en"))
    cross_validate(dataset, with_scaler(make_nn()[0], "nn"))


def test_simple(dataset, num_features):
    selector = sklearn.feature_selection.SelectKBest(score_func=sklearn.feature_selection.f_regression, k=num_features)

    # select from predict sum
    selector.fit(dataset.inputs, dataset.outputs.sum(axis=1))

    reduced_dataset = dataset.select_features(num_features, selector.scores_)

    # try a couple simple models
    test_models(reduced_dataset, "select from predicting mean")


def test_simple_multivariate(dataset, num_features):
    selector = sklearn.feature_selection.SelectKBest(score_func=sklearn.feature_selection.f_regression, k=num_features)

    scores = None

    for output_index in range(dataset.outputs.shape[1]):
        output = dataset.outputs[:, output_index]
        selector.fit(dataset.inputs, output)

        if scores is None:
            scores = selector.scores_.copy()
        else:
            scores += selector.scores_

    reduced_dataset = dataset.select_features(num_features, scores)

    # try a couple simple models
    test_models(reduced_dataset, "select from predicting each output")


def test_select_from_en(dataset, num_features):
    model = with_scaler(sklearn.linear_model.ElasticNet(0.001), "en")
    model.fit(dataset.inputs, dataset.outputs.sum(axis=1))

    reduced_dataset = dataset.select_features(num_features, abs(model.named_steps["en"].coef_), verbose=1)
    test_models(reduced_dataset, "select from elastic net predicting sum")


def test_select_from_rf(dataset, num_features):
    model = sklearn.ensemble.RandomForestRegressor(40, min_samples_split=20)
    model.fit(dataset.inputs, dataset.outputs.sum(axis=1))

    reduced_dataset = dataset.select_features(num_features, model.feature_importances_, verbose=1)
    test_models(reduced_dataset, "select from random forest")


def test_rfecv_en(dataset, num_features):
    model = sklearn.linear_model.ElasticNet(0.001)
    selector = sklearn.feature_selection.RFECV(model, cv=3, scoring=rms_error)

    X = helpers.sk.ClippedRobustScaler().fit_transform(dataset.inputs)
    selector.fit(X, dataset.outputs.mean(axis=1))

    reduced_dataset = dataset.select_features(num_features, selector.ranking_, higher_is_better=False, verbose=1)
    test_models(reduced_dataset, "select from elastic net predicting sum")


def test_subspace_selection(dataset, num_features):
    orig_num_features = dataset.inputs.shape[1]
    while dataset.inputs.shape[1] > num_features * 2:
        model = helpers.sk.MultivariateBaggingRegressor(with_scaler(sklearn.linear_model.Ridge(), "rr"), max_features=num_features, n_estimators=orig_num_features)
        feature_scores = model.evaluate_features_cv(dataset.inputs, dataset.outputs, dataset.splits)
        dataset = dataset.select_features(dataset.inputs.shape[1] / 2, feature_scores, higher_is_better=False)

    model = helpers.sk.MultivariateBaggingRegressor(with_scaler(sklearn.linear_model.Ridge(), "rr"), max_features=num_features, n_estimators=orig_num_features)
    feature_scores = model.evaluate_features_cv(dataset.inputs, dataset.outputs, dataset.splits)
    dataset = dataset.select_features(num_features, feature_scores, higher_is_better=False)

    test_models(dataset, "random subset method")




def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    dataset = load_split_data(args)

    print("Baselines")
    cross_validate(dataset, sklearn.dummy.DummyRegressor())
    test_models(dataset, "baseline")

    test_simple_multivariate(dataset, args.num_features)
    test_simple(dataset, args.num_features)

    # select from elasticnet
    test_select_from_en(dataset, args.num_features)
    test_select_from_rf(dataset, args.num_features)
    test_rfecv_en(dataset, args.num_features)
    test_subspace_selection(dataset, args.num_features)

    # rfe on elasticnet

    # my rfe

    # my new method

if __name__ == "__main__":
    sys.exit(main())
