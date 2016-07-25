from __future__ import print_function
from __future__ import unicode_literals

import argparse
import io

import collections
import logging

import numpy
import pandas
import sys

from helpers.general import with_num_features, with_date, _with_extra
from helpers.sk import with_model_name, predictions_in_training_range
from loaders import separate_output, load_data
from train_test import make_nn, make_rnn, make_blr, make_rf
import helpers.sk
import sklearn.linear_model
import os.path

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--resample", default="1H", help="Time interval to resample the training data")
    parser.add_argument("--model", default="nn", help="Model to generate predictions with")
    parser.add_argument("--graph-dir", default=None, help="Generate graphs of predictions and learning curves into this dir")
    parser.add_argument("--clip", default=None, help="Json file with saved output clip min and max values")
    parser.add_argument("training_dir", help="Dir with the training CSV files")
    parser.add_argument("testing_dir", help="Dir with the testing files, including the empty prediction file")
    parser.add_argument("prediction_file", help="Destination for predictions")

    return parser.parse_args()


def get_model(model_name):
    model_name = model_name.lower()

    if model_name in {"nn", "mlp"}:
        return make_nn(history_file="nn_learning.csv")[0]
    elif model_name == "rnn":
        return make_rnn(history_file="rnn_learning.csv")[0]
    elif model_name == "rnn_relu":
        return make_rnn(history_file="rnn_learning.csv", non_negative=True)[0]
    elif model_name == "rnnx2":
        base_model = make_rnn(history_file="rnn_learning.csv", time_steps=12)[0]
        return helpers.sk.AverageClonedRegressor(base_model, 2)
    elif model_name == "blr":
        return make_blr()
    elif model_name in {"elastic", "elasticnet", "en"}:
        return sklearn.linear_model.ElasticNet(0.01)
    elif model_name in {"rf", "randomforest", "forest"}:
        return make_rf()
    else:
        raise ValueError("Unknown model abbreviation '{}'".format(model_name))

def with_resample(filename, resample_interval):
    return _with_extra(filename, "resample_{}".format(resample_interval.lower()))

def graph_predictions(X_test, baseline_model, model, Y_train, output_file, test_index):
    num_outputs = 5

    important_columns = [c for c, _ in rate_columns(Y_train).most_common(num_outputs)]
    output_names = Y_train.columns

    Y_baseline = pandas.DataFrame(baseline_model.predict(X_test), columns=output_names, index=test_index)
    Y_baseline = Y_baseline[important_columns]
    Y_model = pandas.DataFrame(model.predict(X_test), columns=output_names, index=test_index)
    Y_model = Y_model[important_columns]

    # dummy resample for predict-mean
    resample = "1D"

    axes = Y_baseline.resample(resample).mean().plot(figsize=(16, 9), ylim=(0, 2))
    axes.set_title("Predictions of top mean+std outputs resampled to {}".format(resample))
    axes.get_figure().savefig(with_resample(with_model_name(output_file, baseline_model, snake_case=True), resample), dpi=300)

    for resample in "1H 6H 1D".split():
        axes = Y_model.resample(resample).mean().plot(figsize=(16, 9), ylim=(0, 2))
        axes.set_title("Predictions of top mean+std outputs resampled to {}".format(resample))
        axes.get_figure().savefig(with_resample(with_model_name(output_file, model, snake_case=True), resample), dpi=300)


def rate_columns(data):
    # TODO: This is a candidate for shared lib
    """Rate columns by mean and stddev"""
    return collections.Counter({c: data[c].mean() + data[c].std() for c in data.columns})

def get_history(model):
    try:
        return model.history_df_
    except AttributeError:
        return get_history(model.estimator_)


def save_history(model, output_file):
    try:
        history = get_history(model)
        axes = history.plot(figsize=(16, 9), logy=True)
        axes.get_figure().savefig(with_model_name(output_file, model, snake_case=True), dpi=300)
    except AttributeError:
        print("Not saving model learning curve graph cause it doesn't exist")


def predict_test_data(X_train, Y_train, args):
    # retrain baseline model as a sanity check
    baseline_model = sklearn.dummy.DummyRegressor("mean").fit(X_train, Y_train)

    # retrain a model on the full data
    model = get_model(args.model)

    if args.clip:
        # load the clipper
        with io.open(args.clip, "rb") as json_in:
            clipper = helpers.sk.OutputClippedTransform.load(json_in)
            # print("Loaded clipper:", clipper.max, clipper.min)

        # wrap the model with the clipper
        model = helpers.sk.OutputTransformation(model, clipper)

    model = model.fit(X_train, Y_train.values)

    test_data = load_data(args.testing_dir)
    X_test, Y_test = separate_output(test_data)

    Y_pred = model.predict(X_test)
    test_data[Y_train.columns] = Y_pred

    verify_predictions(X_test, baseline_model, model)
    print("Percent of predictions in training data range: {:.2f}%".format(100. * predictions_in_training_range(Y_train, Y_pred)))

    if args.graph_dir:
        import matplotlib
        matplotlib.use("Agg")
        save_history(model, os.path.join(args.graph_dir, "learning_curve.png"))
        graph_predictions(X_test, baseline_model, model, Y_train, os.path.join(args.graph_dir, "predictions.png"), test_data.index)

    # redo the index as unix timestamp
    test_data.index = test_data.index.astype(numpy.int64) / 10 ** 6
    test_data[Y_test.columns].to_csv(with_date(with_model_name(with_num_features(args.prediction_file, X_train), model, snake_case=True)), index_label="ut_ms")


def verify_predictions(X_test, baseline_model, model):
    Y_pred_baseline = baseline_model.predict(X_test)
    Y_pred = model.predict(X_test)

    deltas = numpy.abs(Y_pred - Y_pred_baseline) / numpy.abs(Y_pred_baseline)
    per_row = deltas.mean()

    unusual_rows = ~(per_row < 5)
    unusual_count = unusual_rows.sum()
    if unusual_count > 0:
        print("{:.1f}% ({:,} / {:,}) of rows have unusual predictions:".format(100. * unusual_count / Y_pred.shape[0], unusual_count, Y_pred.shape[0]))

        unusual_inputs = X_test[unusual_rows].reshape(-1, X_test.shape[1])
        unusual_outputs = Y_pred[unusual_rows].reshape(-1, Y_pred.shape[1])

        for i in range(unusual_inputs.shape[0]):
            print(("Input: ", unusual_inputs[i]))
            print(("Output: ", unusual_outputs[i]))

    overall_delta = per_row.mean()
    print("Percent change from baseline: {:.2f}%".format(100. * overall_delta))
    print("Percent predictions below zero: {:.1f}%".format(100 * (Y_pred < 0).mean()))


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    train_data = load_data(args.training_dir, resample_interval=args.resample, filter_null_power=True)

    X_train, Y_train = separate_output(train_data)

    predict_test_data(X_train, Y_train, args)


if __name__ == "__main__":
    sys.exit(main())