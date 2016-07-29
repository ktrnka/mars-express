from __future__ import print_function
from __future__ import unicode_literals

import argparse
import collections
import logging
import sys
from pprint import pprint

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
from helpers.sk import rms_error
from loaders import load_split_data


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--debug", default=False, action="store_true", help="Debug logging")
    parser.add_argument("-q", "--quieter", default=False, action="store_true", help="Don't show info logging messages")

    parser.add_argument("--time-steps", default=4, type=int, help="Number of time steps for recurrent/etc models")
    parser.add_argument("--verify", default=False, action="store_true", help="Run verifications on the input data for outliers and such")

    parser.add_argument("--feature-pairs", default=False, action="store_true", help="Try out pairs of features")
    parser.add_argument("--resample", default="1H", help="Time interval to resample the training data")
    parser.add_argument("--extra-analysis", default=False, action="store_true", help="Extra analysis on the data")
    parser.add_argument("--analyse-feature-importance", default=False, action="store_true", help="Analyse feature importance and print them out for some models")
    parser.add_argument("--analyse-hyperparameters", default=False, action="store_true", help="Analyse hyperparameters and print them out for some models")

    parser.add_argument("training_dir", help="Dir with the training CSV files or joined CSV file with the complete feature matrix")
    return parser.parse_args()


def make_nn(history_file=None, **kwargs):
    """Make a plain neural network with reasonable default args"""

    model = helpers.neural.NnRegressor(num_epochs=500,
                                       batch_size=256,
                                       learning_rate=0.004,
                                       dropout=0.5,
                                       activation="elu",
                                       input_noise=0.05,
                                       input_dropout=0.02,
                                       hidden_units=200,
                                       early_stopping=True,
                                       l2=0.0001,
                                       val=.1,
                                       maxnorm=True,
                                       history_file=history_file,
                                       lr_decay=0.99,
                                       non_negative=True,
                                       assert_finite=False,
                                       **kwargs)

    model = with_append_mean(model)
    model = with_scaler(model, "nn")

    prefix = "nn__estimator__"

    return model, prefix

@helpers.general.Timed
def experiment_neural_network(dataset, tune_params=False):
    model, param_prefix = make_nn()
    cross_validate(dataset, model)

    if tune_params:
        print("Running hyperparam opt")
        nn_hyperparams = {
            "learning_rate": helpers.sk.RandomizedSearchCV.exponential(1e-2, 1e-4),
            "lr_decay": helpers.sk.RandomizedSearchCV.exponential(1 - 1e-2, 1 - 1e-5),
            # "input_dropout": helpers.sk.RandomizedSearchCV.uniform(0., 0.1),
            "input_noise": helpers.sk.RandomizedSearchCV.uniform(0.05, 0.2),
            "hidden_units": helpers.sk.RandomizedSearchCV.uniform(100, 500),
            "dropout": helpers.sk.RandomizedSearchCV.uniform(0.3, 0.7)
        }
        nn_hyperparams = {param_prefix + k: v for k, v in nn_hyperparams.items()}
        model, _ = make_nn()
        model.history_file = None
        wrapped_model = helpers.sk.RandomizedSearchCV(model, nn_hyperparams, n_iter=20, scoring=rms_error, cv=dataset.splits, refit=False)
        # cross_validate(X_train, Y_train, wrapped_model, "RandomizedSearchCV(NnRegressor)", splits)

        wrapped_model.fit(dataset.inputs, dataset.outputs)
        wrapped_model.print_tuning_scores()


def make_rnn(history_file=None, time_steps=4, non_negative=False, early_stopping=True, reverse=False):
    """Make a recurrent neural network with reasonable default args for this task"""
    model = helpers.neural.RnnRegressor(learning_rate=7e-4,
                                        num_units=50,
                                        time_steps=time_steps,
                                        batch_size=64,
                                        num_epochs=300,
                                        verbose=0,
                                        input_noise=0.05,
                                        input_dropout=0.02,
                                        early_stopping=early_stopping,
                                        recurrent_dropout=0.65,
                                        dropout=0.5,
                                        val=0.1,
                                        assert_finite=False,
                                        pretrain=True,
                                        non_negative=non_negative,
                                        reverse=reverse,
                                        history_file=history_file)

    model = with_append_mean(model)
    model = with_scaler(model, "rnn")

    prefix = "rnn__estimator__"

    return model, prefix


def with_append_mean(model):
    """Force the model to also predict the sum of outputs"""
    return helpers.sk.OutputTransformation(model, helpers.sk.QuickTransform.make_append_mean())


def with_non_negative(model):
    """Wrap the model in another model that forces outputs to be positive"""
    return helpers.sk.OutputTransformation(model, helpers.sk.QuickTransform.make_non_negative())


@helpers.general.Timed
def experiment_rnn(dataset, tune_params=False, time_steps=4):
    model, param_prefix = make_rnn(time_steps=time_steps)
    cross_validate(dataset, model)

    if tune_params:
        hyperparams = {
            "learning_rate": helpers.sk.RandomizedSearchCV.uniform(5e-3, 5e-4),
            # "lr_decay": [0.999, 1],
            # "num_units": [25, 50, 100],
            # "dropout": helpers.sk.RandomizedSearchCV.uniform(0.35, 0.65),
            "recurrent_dropout": helpers.sk.RandomizedSearchCV.uniform(0.4, 0.7),
            "time_steps": [8],
            # "input_dropout": [0.02, 0.04],
        }
        hyperparams = {param_prefix + k: v for k, v in hyperparams.items()}

        wrapped_model = helpers.sk.RandomizedSearchCV(model, hyperparams, n_iter=8, n_jobs=1, scoring=rms_error, refit=False, cv=dataset.splits)

        wrapped_model.fit(dataset.inputs, dataset.outputs)
        wrapped_model.print_tuning_scores()


def main():
    args = parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quieter:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO)

    dataset = load_split_data(args, data_loader=load_data_fixed)

    baseline_model = sklearn.dummy.DummyRegressor("mean")
    cross_validate(dataset, baseline_model)

    model = with_scaler(sklearn.linear_model.LinearRegression(), "lr")
    cross_validate(dataset, model)

    experiment_elastic_net(dataset, feature_importance=True)

    experiment_neural_network(dataset, tune_params=False and args.analyse_hyperparameters)

    experiment_rnn(dataset, tune_params=True and args.analyse_hyperparameters, time_steps=args.time_steps)


def experiment_elastic_net(dataset, feature_importance=True):
    model = with_scaler(sklearn.linear_model.ElasticNet(0.001), "en")
    cross_validate(dataset, model)

    if feature_importance:
        model.fit(dataset.inputs, dataset.outputs)

        feature_importances = collections.Counter()
        for fname, fweight in zip(dataset.feature_names, helpers.sk.get_lr_importances(model.named_steps["en"])):
            feature_importances[fname] = fweight
        print("Feature potentials from ElasticNet (max of abs per-output coefs)")
        pprint(feature_importances.most_common())


def with_scaler(model, name):
    return sklearn.pipeline.Pipeline([("scaler", helpers.sk.ClippedRobustScaler()), (name, model)])


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


def make_rf():
    """Make a random forest model with reasonable default args"""
    return sklearn.ensemble.RandomForestRegressor(25, min_samples_leaf=35, max_depth=34, max_features=20)


def cross_validate(dataset, model, n_jobs=1):
    if not dataset.split_map:
        scores = sklearn.cross_validation.cross_val_score(model, dataset.inputs, dataset.outputs, scoring=rms_error, cv=dataset.splits, n_jobs=n_jobs)
        print("{}: {:.4f} +/- {:.4f}".format(helpers.sk.get_model_name(model), -scores.mean(), scores.std()))
        helpers.general.get_function_logger().info("{}: {:.4f} +/- {:.4f}".format(helpers.sk.get_model_name(model), -scores.mean(), scores.std()))
    else:
        for split_name, split in dataset.split_map.items():
            scores = sklearn.cross_validation.cross_val_score(model, dataset.inputs, dataset.outputs, scoring=rms_error, cv=split, n_jobs=n_jobs)
            print("[CV={}] {}: {:.4f} +/- {:.4f}".format(split_name, helpers.sk.get_model_name(model), -scores.mean(), scores.std()))
            helpers.general.get_function_logger().info("[CV={}] {}: {:.4f} +/- {:.4f}".format(split_name, helpers.sk.get_model_name(model), -scores.mean(), scores.std()))


if __name__ == "__main__":
    sys.exit(main())