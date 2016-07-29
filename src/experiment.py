from __future__ import print_function
from __future__ import unicode_literals

import argparse
import collections
import logging
import sys
from operator import itemgetter

import sklearn

import helpers.general
import helpers.neural
import helpers.sk
from helpers.sk import rms_error, with_val
from loaders import find_files, load_series, load_split_data, add_loader_parse, get_loader
from train_test import make_nn, cross_validate, make_rnn, with_non_negative, with_scaler, with_append_mean

"""
Dumping ground for one-off experiments so that they don't clog up train_test so much.
"""

N_JOBS = -1

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verify", default=False, action="store_true", help="Run verifications on the input data for outliers and such")
    parser.add_argument("--resample", default="1H", help="Time interval to resample the training data")
    parser.add_argument("--extra-analysis", default=False, action="store_true", help="Extra analysis on the data")
    parser.add_argument("--split", default="alex", choices=["timecv", "years", "alex"])

    parser.add_argument("training_dir", help="Dir with the training CSV files or joined CSV file with the complete feature matrix")
    add_loader_parse(parser)
    return parser.parse_args()


def test_mlp_ensembles(dataset):
    print("MLP baseline")
    model, _ = make_nn()
    cross_validate(dataset, model)

    print("MLP x 10")
    model, _ = make_nn()
    cross_validate(dataset, helpers.sk.AverageClonedRegressor(model, 10))

    print("MLP x 10 @ 90% features")
    model, _ = make_nn()
    cross_validate(dataset, helpers.sk.MultivariateBaggingRegressor(model, n_estimators=10, max_features=0.9))


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    dataset = load_split_data(args, data_loader=get_loader(args), split_type=args.split)
    dataset.split_map = None

    test_rnn_params_last_ditch_effort(dataset)

def test_features(dataset):
    from helpers.features import rfe_slow

    model = sklearn.linear_model.LinearRegression()
    rfe_slow(dataset, model, rms_error)


def test_ewma_output(dataset):
    print("MLP with EWMA")
    model, _ = make_nn()
    model = helpers.sk.OutputTransformation(model, helpers.sk.QuickTransform.make_ewma_outputs(num_spans=1))
    cross_validate(dataset, model)

    print("MLP")
    model, _ = make_nn()
    cross_validate(dataset, model)

def test_tree_methods(dataset):
    import sklearn.ensemble

    model = sklearn.ensemble.RandomForestRegressor(200, max_depth=30, min_samples_split=20, n_jobs=N_JOBS)
    cross_validate(dataset, model)

def test_skflow(dataset):
    import tensorflow.contrib.learn as skflow
    import tensorflow as tf

    batch_size = 256
    s2e = dataset.inputs.shape[0] / float(batch_size)

    def decay_function(global_step):
        return tf.train.exponential_decay(learning_rate=0.004, global_step=global_step, decay_steps=int(s2e), decay_rate=0.99, staircase=True)

    print("TensorFlow Learn MLP")
    model = skflow.TensorFlowDNNRegressor(hidden_units=[200], batch_size=batch_size, steps=int(s2e * 500), optimizer="Adam", dropout=0.5, learning_rate=decay_function)
    cross_validate(dataset, model)

    print("MLP")
    cross_validate(dataset, make_nn()[0])

@helpers.general.Timed
def tune_random_forest(dataset):
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=500)
    cross_validate(dataset, model)

    hyperparams = {
        "max_features": range(int(0.5 * dataset.inputs.shape[1]), dataset.inputs.shape[1] + 1),
        "max_depth": range(10, 150),
        "min_samples_split": range(2, 40)
    }

    wrapped_model = helpers.sk.RandomizedSearchCV(model, hyperparams, n_iter=50, n_jobs=N_JOBS, scoring=rms_error, refit=True)

    wrapped_model.fit(dataset.inputs.copy(), dataset.outputs.copy())
    wrapped_model.print_tuning_scores()

    cross_validate(dataset, wrapped_model)



@helpers.general.Timed
def tune_gradient_boosting(dataset):
    base_model = sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, subsample=0.9, learning_rate=0.26, min_samples_split=20, max_depth=4, random_state=4)
    model = helpers.sk.MultivariateRegressionWrapper(base_model)
    cross_validate(dataset, model)

    hyperparams = {
        "learning_rate": helpers.sk.RandomizedSearchCV.uniform(0.1, 0.5),
        "max_depth": range(3, 6),
        "min_samples_split": range(2, 40),
        "subsample": [0.9],
        "max_features": range(int(0.8 * dataset.inputs.shape[1]), dataset.inputs.shape[1] + 1),
        # "loss": ["ls", "lad", "huber", "quantile"]
    }

    # TODO: Up the number of hyperparameter samples greatly
    wrapped_model = helpers.sk.MultivariateRegressionWrapper(sklearn.grid_search.RandomizedSearchCV(base_model, hyperparams, n_iter=10, cv=4, refit=True, n_jobs=N_JOBS, scoring=rms_error))

    wrapped_model.fit(dataset.inputs.copy(), dataset.outputs.copy())
    wrapped_model.print_best_params()

    cross_validate(dataset, wrapped_model)

@helpers.general.Timed
def test_new_gradient_boosting(dataset):
    base_model = sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, min_samples_split=20, max_depth=9, max_features=90, random_state=4)
    model = helpers.sk.JoinedMultivariateRegressionWrapper(base_model)
    cross_validate(dataset, model)

    # hyperparam tuning is slow so just pick 2-3
    base_model = sklearn.ensemble.GradientBoostingRegressor(n_estimators=200, learning_rate=0.06, min_samples_split=20, max_depth=9, max_features=90, random_state=4)
    model = helpers.sk.JoinedMultivariateRegressionWrapper(base_model)
    cross_validate(dataset, model)

    # hyperparam tuning is slow so just pick 2-3
    base_model = sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, min_samples_split=20, max_depth=9, max_features=90, random_state=4)
    model = helpers.sk.JoinedMultivariateRegressionWrapper(base_model)
    cross_validate(dataset, model)

    # hyperparams = {
    #     "learning_rate": helpers.sk.RandomizedSearchCV.uniform(0.05, 0.15),
    #     "max_depth": range(7, 12),
    #     "min_samples_split": range(2, 40),
    #     # "subsample": [0.9],
    #     "max_features": range(int(0.8 * dataset.inputs.shape[1]), dataset.inputs.shape[1] + 1),
    #     # "loss": ["ls", "lad", "huber", "quantile"]
    # }
    #
    # wrapped_model = helpers.sk.JoinedMultivariateRegressionWrapper(helpers.sk.RandomizedSearchCV(base_model, hyperparams, n_iter=10, refit=True, cv=4, n_jobs=N_JOBS, scoring=rms_error))
    # wrapped_model.fit(dataset.inputs.copy(), dataset.outputs.copy())
    # wrapped_model.estimator_.print_tuning_scores()
    #
    # cross_validate(dataset, wrapped_model)


def test_rnn_relu(dataset):
    print("RNN with ReLU")
    model, _ = make_rnn()
    model.estimator.non_negative = True
    cross_validate(dataset, model)

    print("RNN base with nonneg clipper")
    model = with_non_negative(make_rnn()[0])
    cross_validate(dataset, model)


def test_input_noise(dataset):
    print("RNN with half input noise 2%")
    model, _ = make_rnn()
    model.estimator.input_noise = 0.02
    cross_validate(dataset, model)

    print("RNN with half input noise 1%")
    model, _ = make_rnn()
    model.estimator.input_noise = 0.01
    cross_validate(dataset, model)

    print("RNN base 5% now")
    model, _ = make_rnn()
    cross_validate(dataset, model)


def test_rnn_l2(dataset):
    print("RNN with small L2")
    model, _ = make_rnn()
    model.estimator.l2 = 1e-6
    cross_validate(dataset, model)

    print("RNN base")
    model, _ = make_rnn()
    cross_validate(dataset, model)


def test_mlp_no_val(dataset):
    print("MLP baseline")
    model, _ = make_nn()
    cross_validate(dataset, model)

    print("MLP no validation")
    model, _ = make_nn()
    model.estimator.val = 0.
    cross_validate(dataset, model)


def test_rnn_smaller_batches(dataset):
    print("RNN reduced batch size")
    model, _ = make_rnn()
    model.estimator.batch_size = 64
    cross_validate(dataset, model)

    print("RNN baseline")
    model, _ = make_rnn()
    cross_validate(dataset, model)


def test_mlp_sample_weight(dataset):
    print("MLP with ReLU and sample weight")
    model, _ = make_nn(weight_samples=True)
    cross_validate(dataset, model)

    print("MLP base")
    model, _ = make_nn()
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
    model, prefix = make_rnn(non_negative=True)
    # model.set_params(**{prefix + "val": 0.1})

    # base = no schedule
    print("Baseline RNN")
    # model.set_params(**{prefix + "history_file": "nn_default.csv"})
    cross_validate(dataset, model)
    # model.fit(dataset.inputs, dataset.outputs)

    # higher init, decay set to reach the same at 40 epochs
    print("RNN with decay")
    model.set_params(**{prefix + "lr_decay": "DecreasingLearningRateScheduler"})
    # model.fit(dataset.inputs, dataset.outputs)
    cross_validate(dataset, model)


def test_rnn_no_early_stop(dataset):
    """Try a few different learning rate schedules"""
    model, prefix = make_rnn()

    print("RNN without early stopping")
    cross_validate(dataset, model)

    print("RNN without early stopping, run 2")
    cross_validate(dataset, model)



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

    wrapped_model = helpers.sk.RandomizedSearchCV(model, hyperparams, n_iter=10, n_jobs=N_JOBS, scoring=rms_error,
                                                  cv=dataset.splits, refit=False)

    wrapped_model.fit(dataset.inputs, dataset.outputs)
    wrapped_model.print_tuning_scores()


def test_clones(dataset, n=2):
    print("Baseline model")
    cross_validate(dataset, make_rnn()[0])

    print("Ensemble of {}".format(n))
    model = helpers.sk.AverageClonedRegressor(make_rnn()[0], n)
    cross_validate(dataset, model)


def test_stateful_rnn(dataset):
    model, _ = make_rnn()
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


def test_stacking(dataset):
    # build the component models
    models = [with_val(with_scaler(sklearn.linear_model.ElasticNet(0.001), "en")),
              with_val(with_scaler(sklearn.linear_model.RidgeCV(), "ridge")),
              make_nn()[0]]
    for model in models:
        cross_validate(dataset, model)

    print("Ridge stacking")
    ensemble = helpers.sk.StackedEnsembleRegressor(models, with_non_negative(sklearn.linear_model.Ridge(alpha=0.1, fit_intercept=False)))
    cross_validate(dataset, ensemble)

    print("Ridge/tuned stacking")
    hyperparams = {
        "fit_intercept": [True, False],
        "alpha": helpers.sk.RandomizedSearchCV.exponential(1, 1e-5),
        "normalize": [True, False]
    }
    wrapped_model = helpers.sk.RandomizedSearchCV(sklearn.linear_model.Ridge(), hyperparams, n_iter=10, scoring=rms_error, cv=3, refit=True)
    ensemble = helpers.sk.StackedEnsembleRegressor(models, with_non_negative(wrapped_model))
    cross_validate(dataset, ensemble)

    print("ElasticNet stacking")
    ensemble = helpers.sk.StackedEnsembleRegressor(models, with_non_negative(sklearn.linear_model.ElasticNet(0.001, fit_intercept=False)))
    cross_validate(dataset, ensemble)

    print("ElasticNet/tuned stacking")
    hyperparams = {
        "fit_intercept": [True, False],
        "alpha": helpers.sk.RandomizedSearchCV.exponential(1, 1e-5),
        "normalize": [True, False],
        "l1_ratio": helpers.sk.RandomizedSearchCV.uniform(.25, .75)
    }
    wrapped_model = helpers.sk.RandomizedSearchCV(sklearn.linear_model.ElasticNet(), hyperparams, n_iter=10, scoring=rms_error, cv=3, refit=True)
    ensemble = helpers.sk.StackedEnsembleRegressor(models, with_non_negative(wrapped_model))
    cross_validate(dataset, ensemble)

    print("ElasticNet/tuned stacking more aggressive")
    hyperparams = {
        "fit_intercept": [True, False],
        "alpha": helpers.sk.RandomizedSearchCV.exponential(1, 1e-5),
        "normalize": [True, False],
        "l1_ratio": helpers.sk.RandomizedSearchCV.uniform(.1, .9)
    }
    wrapped_model = helpers.sk.RandomizedSearchCV(sklearn.linear_model.ElasticNet(), hyperparams, n_iter=40, scoring=rms_error, cv=3, refit=True)
    ensemble = helpers.sk.StackedEnsembleRegressor(models, with_non_negative(wrapped_model))
    cross_validate(dataset, ensemble)

    # now do a test that's stacking MLP, forwards, backwards RNN
    models = [with_non_negative(make_rnn(time_steps=4)[0]),
              with_non_negative(make_rnn(time_steps=8)[0]),
              with_val(with_non_negative(make_rnn(time_steps=4, reverse=True)[0])),
              make_nn()[0]]
    for model in models:
        cross_validate(dataset, model)

    print("Ridge stacking")
    ensemble = helpers.sk.StackedEnsembleRegressor(models, with_non_negative(sklearn.linear_model.Ridge(alpha=0.1, fit_intercept=False)))
    cross_validate(dataset, ensemble)

    print("Ridge/tuned stacking")
    hyperparams = {
        "fit_intercept": [True, False],
        "alpha": helpers.sk.RandomizedSearchCV.exponential(1, 1e-5),
        "normalize": [True, False]
    }
    wrapped_model = helpers.sk.RandomizedSearchCV(sklearn.linear_model.Ridge(), hyperparams, n_iter=10, scoring=rms_error, cv=3, refit=True)
    ensemble = helpers.sk.StackedEnsembleRegressor(models, with_non_negative(wrapped_model))
    cross_validate(dataset, ensemble)

    print("ElasticNet stacking")
    ensemble = helpers.sk.StackedEnsembleRegressor(models, with_non_negative(sklearn.linear_model.ElasticNet(0.001, fit_intercept=False)))
    cross_validate(dataset, ensemble)

    print("ElasticNet/tuned stacking")
    hyperparams = {
        "fit_intercept": [True, False],
        "alpha": helpers.sk.RandomizedSearchCV.exponential(1, 1e-5),
        "normalize": [True, False],
        "l1_ratio": helpers.sk.RandomizedSearchCV.uniform(.25, .75)
    }
    wrapped_model = helpers.sk.RandomizedSearchCV(sklearn.linear_model.ElasticNet(), hyperparams, n_iter=30, scoring=rms_error, cv=3, refit=True)
    ensemble = helpers.sk.StackedEnsembleRegressor(models, with_non_negative(wrapped_model))
    cross_validate(dataset, ensemble)

def test_rnn_params_last_ditch_effort(dataset):
    base_model = with_non_negative(make_rnn(time_steps=4)[0])

    model2 = helpers.neural.RnnRegressor(learning_rate=7e-3,
                                         lr_decay="DecreasingLearningRateScheduler",
                                         clip_gradient_norm=5,
                                         num_units=50,
                                         time_steps=4,
                                         batch_size=64,
                                         num_epochs=300,
                                         verbose=0,
                                         input_noise=0.03,
                                         input_dropout=0.01,
                                         early_stopping=True,
                                         recurrent_dropout=0.1,
                                         dropout=0.5,
                                         val=0.1,
                                         assert_finite=False,
                                         pretrain=True,
                                         non_negative=True,
                                         reverse=False)

    model2 = with_append_mean(model2)
    model2 = with_scaler(model2, "rnn")

    for _ in range(3):
        print("All tweaks that seem interesting to me")
        cross_validate(dataset, model2)

        print("Baseline RNN")
        cross_validate(dataset, base_model)



def test_reversed_rnn(dataset):
    print("Base MLP")
    cross_validate(dataset, with_non_negative(make_nn()[0]))

    print("Base RNN")
    base_model = with_non_negative(make_rnn(time_steps=4)[0])
    cross_validate(dataset, base_model)

    print("Reversed RNN")
    base_model = with_non_negative(make_rnn(time_steps=4, reverse=True)[0])
    cross_validate(dataset, base_model)

    # try ensemble of the two RNNs
    forward_model = with_non_negative(make_rnn(time_steps=4)[0])
    backward_model = with_non_negative(make_rnn(time_steps=4, reverse=True)[0])

    for forward_weight in [0.5, 0.6, 0.8, 0.9]:
        print("Ensemble RNN @ {} forwards".format(forward_weight))
        ensemble = helpers.sk.WeightedEnsembleRegressor([forward_model, backward_model], weights=[forward_weight, 1 - forward_weight])
        cross_validate(dataset, ensemble)


def test_realistic_rnns(dataset, num_clones=2):
    # print("Base regular")
    # base_model = with_non_negative(make_rnn(time_steps=4))
    # cross_validate(dataset, base_model)

    for time_steps in [4, 8]:
        print("Time={} RNNx{}".format(time_steps, num_clones))
        base_model = with_non_negative(make_rnn(time_steps=time_steps)[0])
        ensembled_model = helpers.sk.AverageClonedRegressor(base_model, num_clones)
        cross_validate(dataset, ensembled_model)


def test_time_onestep(dataset):
    # base estimator
    model, _ = make_nn()

    print("Baseline")
    cross_validate(dataset, model)

    time_model = helpers.sk.DeltaSumRegressor(model, num_rolls=2)
    cross_validate(dataset, time_model)


def test_output_augmentations(dataset):
    base_model, _ = make_rnn()
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
