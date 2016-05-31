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
    test_models(reduced_dataset, "f_regression(sum)")

    # try with ewma
    reduced_dataset = dataset.select_features(num_features, make_select_f(num_features, True)(dataset.inputs, dataset.outputs.sum(axis=1)))
    test_models(reduced_dataset, "f_regression_ewma(sum)")


def test_simple_multivariate(dataset, num_features):
    scores = multivariate_select(dataset, make_select_f(num_features))
    reduced_dataset = dataset.select_features(num_features, scores)
    test_models(reduced_dataset, "f_regression(multivariate)")

    scores = multivariate_select(dataset, make_select_f(num_features), weight_outputs=True)
    print(scores.shape)
    reduced_dataset = dataset.select_features(num_features, scores)
    test_models(reduced_dataset, "f_regression(weighted multivariate)")


def test_select_from_en(dataset, num_features):
    model = with_scaler(sklearn.linear_model.ElasticNet(0.001), "en")
    model.fit(dataset.inputs, dataset.outputs.sum(axis=1))

    reduced_dataset = dataset.select_features(num_features, abs(model.named_steps["en"].coef_), verbose=1)
    test_models(reduced_dataset, "ElasticNet(sum)")


def score_features_elasticnet(X, Y):
    model = with_scaler(sklearn.linear_model.ElasticNet(0.001), "en")
    model.fit(X, Y.sum(axis=1))
    return abs(model.named_steps["en"].coef_)


def score_features_loo(X, Y, splits, scorer, std_dev_weight=-.05):
    model = with_scaler(sklearn.linear_model.Ridge(), "ridge")
    scores = [0 for _ in range(X.shape[1])]
    baseline = sklearn.cross_validation.cross_val_score(model, X, Y, scoring=scorer, cv=splits).mean()
    for i in range(X.shape[1]):
        included = numpy.asarray([j for j in range(X.shape[1]) if j != i])

        cv_scores = sklearn.cross_validation.cross_val_score(model, X[:, included], Y, scoring=scorer, cv=splits)
        scores[i] = baseline - cv_scores.mean() + std_dev_weight * cv_scores.std()

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

    selector = sklearn.feature_selection.RFECV(model, cv=tuning_splits, step=2, scoring=rms_error)

    X = helpers.sk.ClippedRobustScaler().fit_transform(dataset.inputs)
    selector.fit(X, dataset.outputs.mean(axis=1))

    reduced_dataset = dataset.select_features(num_features, selector.ranking_, higher_is_better=False, verbose=1)
    test_models(reduced_dataset, "RFECV(ElasticNet(sum))")

    model.fit(X, dataset.outputs.sum(axis=1))
    ensembled_weights = abs(model.coef_) / selector.ranking_
    reduced_dataset = dataset.select_features(num_features, ensembled_weights, verbose=1)
    test_models(reduced_dataset, "Ens of RFECV+EN(ElasticNet(sum))")

    ensembled_weights = abs(model.coef_) * .9 ** selector.ranking_
    reduced_dataset = dataset.select_features(num_features, ensembled_weights, verbose=1)
    test_models(reduced_dataset, "Ens of RFECV+EN(ElasticNet(sum)) V2")

    # ensemble with leave one in
    loi_scores = score_features_loi(dataset.inputs, dataset.outputs, tuning_splits, rms_error)
    ensembled_weights = .9 ** loi_scores * abs(model.coef_) * .99 ** selector.ranking_
    reduced_dataset = dataset.select_features(num_features, ensembled_weights, verbose=1)
    test_models(reduced_dataset, "Ens of RFECV+EN+LOI(ElasticNet(sum)) V2")


def test_subspace_selection(dataset, num_features, splits, prefilter=True):
    if prefilter:
        dataset = select_nonzero(dataset)

    orig_num_features = dataset.inputs.shape[1]

    while dataset.inputs.shape[1] > num_features * 2:
        model = helpers.sk.MultivariateBaggingRegressor(with_scaler(sklearn.linear_model.Ridge(), "rr"), max_features=16, n_estimators=orig_num_features)
        feature_scores = model.evaluate_features_cv(dataset.inputs, dataset.outputs, splits)
        dataset = dataset.select_features(0.5, feature_scores, higher_is_better=False)

    model = helpers.sk.MultivariateBaggingRegressor(with_scaler(sklearn.linear_model.Ridge(), "rr"), max_features=num_features, n_estimators=orig_num_features)
    feature_scores = model.evaluate_features_cv(dataset.inputs, dataset.outputs, splits)
    dataset = dataset.select_features(num_features, feature_scores, higher_is_better=False)

    test_models(dataset, "subspace elimination")

def test_loo_loi(dataset, num_features, splits):
    test_models(dataset.select_features(num_features, score_features_loi(dataset.inputs, dataset.outputs, splits, rms_error), verbose=1), "leave one in")

    test_models(dataset.select_features(num_features, score_features_loo(dataset.inputs, dataset.outputs, splits, rms_error), verbose=1), "leave one out")

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    dataset = load_split_data(args)

    print("Baselines")
    cross_validate(dataset, sklearn.dummy.DummyRegressor())
    test_models(dataset, "baseline", with_nn=False)

    tuning_splits = sklearn.cross_validation.KFold(dataset.inputs.shape[0], 3, False)

    test_loo_loi(dataset, args.num_features, tuning_splits)

    test_simple_multivariate(dataset, args.num_features)
    test_select_from_cv2(dataset, args.num_features, tuning_splits)
    test_simple(dataset, args.num_features)

    test_select_from_en(dataset, args.num_features)
    test_select_from_en_cv(dataset, args.num_features, tuning_splits)
    # test_select_from_rf(dataset, args.num_features)
    test_rfecv_en(dataset, args.num_features, tuning_splits)
    test_subspace_selection(dataset, args.num_features, tuning_splits)

if __name__ == "__main__":
    sys.exit(main())
