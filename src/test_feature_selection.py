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


def test_simple(dataset, num_features):
    selector = sklearn.feature_selection.SelectKBest(score_func=sklearn.feature_selection.f_regression, k=num_features)

    # select from predict sum
    selector.fit(dataset.inputs, dataset.outputs.sum(axis=1))

    reduced_dataset = dataset.select_features(num_features, selector.scores_)

    # try a couple simple models
    cross_validate(reduced_dataset, sklearn.linear_model.ElasticNet(0.001))
    cross_validate(reduced_dataset, make_nn()[0])


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    dataset = load_split_data(args)

    print("Baselines")
    cross_validate(dataset, sklearn.dummy.DummyRegressor())
    cross_validate(dataset, sklearn.linear_model.ElasticNet(0.001))
    # cross_validate(dataset, make_nn()[0])


    # try play f regression
    test_simple(dataset, args.num_features)

    # select from elasticnet

    # rfe on elasticnet

    # my rfe

    # my new method

if __name__ == "__main__":
    sys.exit(main())
