from __future__ import unicode_literals
from __future__ import print_function

import collections
import logging
import sys
import argparse
from operator import itemgetter

import sklearn

import helpers.general
import helpers.neural
import helpers.sk
from helpers.sk import rms_error
from train_test import make_nn, cross_validate, make_rnn, load_series, find_files, load_split_data

"""
Dumping ground for one-off experiments so that they don't clog up train_test so much.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", default=False, action="store_true", help="Run verifications on the input data for outliers and such")
    parser.add_argument("--resample", default="1H", help="Time interval to resample the training data")
    parser.add_argument("--extra-analysis", default=False, action="store_true", help="Extra analysis on the data")
    parser.add_argument("training_dir", help="Dir with the training CSV files or joined CSV file with the complete feature matrix")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    dataset = load_split_data(args)

    test_rnn_no_val(dataset)


def test_features(dataset):
    from helpers.features import rfe_slow

    model = sklearn.linear_model.LinearRegression()
    rfe_slow(dataset, model, rms_error)


def test_rnn_relu(dataset):
    print("RNN base")
    model = make_rnn()
    cross_validate(dataset, model)

    print("RNN with ReLU")
    model = make_rnn()
    model.non_negative = True
    cross_validate(dataset, model)


def test_mlp_no_val(dataset):
    print("MLP baseline")
    model = make_nn()
    cross_validate(dataset, model)

    print("MLP no validation")
    model = make_nn()
    model.estimator.val = 0.
    cross_validate(dataset, model)


def test_rnn_no_val(dataset):
    print("RNN no pretrain")
    model = make_rnn(augment_output=True)
    model.estimator.pretrain=False
    cross_validate(dataset, model)

    print("RNN baseline")
    model = make_rnn(augment_output=True)
    cross_validate(dataset, model)


def test_mlp_sample_weight(dataset):
    print("MLP with ReLU and sample weight")
    model = make_nn()
    model.estimator.weight_samples = True
    cross_validate(dataset, model)

    print("MLP with ReLU")
    model = make_nn()
    cross_validate(dataset, model)


def test_resample_clipper(training_dir):
    unsampled_outputs = load_series(find_files(training_dir, "power")).dropna()
    unsampled_clipper = helpers.sk.OutputClippedTransform.from_data(unsampled_outputs.values)

    sampled_outputs = unsampled_outputs.resample("5Min").mean().dropna()
    sampled_clipper = helpers.sk.OutputClippedTransform.from_data(sampled_outputs.values)

    for i, output in enumerate(unsampled_outputs.columns):
        print(output)
        print("\tMin: unsampled {}, sampled {} ({:.1f}% higher)".format(unsampled_clipper.min[i], sampled_clipper.min[i], 100. * (sampled_clipper.min[i] - unsampled_clipper.min[i]) / unsampled_clipper.min[i]))
        print("\tMax: unsampled {}, sampled {} ({:.1f}% lower)".format(unsampled_clipper.max[i], sampled_clipper.max[i], 100. * (unsampled_clipper.max[i] - sampled_clipper.max[i]) / unsampled_clipper.max[i]))


def test_schedules(dataset):
    """Try a few different learning rate schedules"""
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


def test_predict_mean(dataset):
    """Try augmenting the output with the mean of outputs to make it easier to learn"""
    model = make_nn()

    cross_validate(dataset, model)
    cross_validate(dataset, helpers.sk.OutputTransformation(model, helpers.sk.QuickTransform.make_append_mean()))


def test_rnn_elu(dataset):
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

    wrapped_model = helpers.sk.RandomizedSearchCV(model, hyperparams, n_iter=10, n_jobs=1, scoring=rms_error,
                                                  cv=dataset.splits, refit=False)

    wrapped_model.fit(dataset.inputs, dataset.outputs)
    wrapped_model.print_tuning_scores()


def test_clones(dataset, n=2):
    print("Baseline model")
    cross_validate(dataset, make_rnn())

    print("Ensemble of {}".format(n))
    model = helpers.sk.AverageClonedRegressor(make_rnn(), n)
    cross_validate(dataset, model)


def test_stateful_rnn(dataset):
    model = make_rnn()
    model.batch_size = 4
    model.time_steps = 4
    model.val = 0
    model.pretrain = False
    model.early_stopping = False
    model.num_epochs = 300

    print("RNN baseline")
    cross_validate(dataset, model)

    print("RNN stateful")
    model.stateful = True
    cross_validate(dataset, model)


def test_realistic_rnns(dataset):
    for time_steps in [4, 8, 12]:
        print("Time={} RNNx2".format(time_steps))
        base_model = make_rnn(augment_output=True, time_steps=time_steps)
        ensembled_model = helpers.sk.AverageClonedRegressor(base_model, 2)
        cross_validate(dataset, ensembled_model)


def test_time_onestep(dataset):
    # base estimator
    model = make_nn()

    print("Baseline")
    cross_validate(dataset, model)

    time_model = helpers.sk.DeltaSumRegressor(model, num_rolls=2)
    cross_validate(dataset, time_model)


def test_output_augmentations(dataset):
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


def score_feature(X_train, Y_train, splits):
    scaler = sklearn.preprocessing.RobustScaler()
    X_train = scaler.fit_transform(X_train.values.reshape(-1, 1))

    model = sklearn.linear_model.LinearRegression()
    return -sklearn.cross_validation.cross_val_score(model, X_train, Y_train, scoring=rms_error, cv=splits).mean()


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

if __name__ == "__main__":
    sys.exit(main())