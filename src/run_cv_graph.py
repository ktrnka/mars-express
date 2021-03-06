from __future__ import unicode_literals
from __future__ import print_function

import collections
import logging
import sys
import argparse

import io
import pandas
import sklearn

import helpers.general
import helpers.sk
from train_test import make_nn
from loaders import load_split_data

"""
Train a few models and recombine their cross-validated predictions then graph the top few at various resampling intervals.
"""


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verify", default=False, action="store_true", help="Run verifications on the input data for outliers and such")
    parser.add_argument("--resample", default="1H", help="Time interval to resample the training data")
    parser.add_argument("--extra-analysis", default=False, action="store_true", help="Extra analysis on the data")
    parser.add_argument("--clip", default=None, help="File with clip values")
    parser.add_argument("training_dir", help="Dir with the training CSV files or joined CSV file with the complete feature matrix")
    parser.add_argument("base_output", help="Base filename for output graphs. The model name and resample interval will be attached.")
    return parser.parse_args()


def cv_graph(dataset, model, graph_filename):
    predictions = sklearn.cross_validation.cross_val_predict(model, dataset.inputs, dataset.outputs, cv=dataset.splits)

    # pick a few outputs
    target_df = pandas.DataFrame(data=dataset.outputs, columns=dataset.target_names, index=dataset.output_index)
    pred_df = pandas.DataFrame(data=predictions, columns=dataset.target_names, index=dataset.output_index)

    scores = collections.Counter({col: target_df[col].mean() + target_df[col].std() for col in target_df.columns})
    cols = [col for col, _ in scores.most_common(5)]

    for resample in ["1H", "6H", "1D"]:
        sampled_df = target_df[cols]
        if resample:
            sampled_df = sampled_df.resample(resample).mean()
        axes = sampled_df.plot(figsize=(16, 9), ylim=(0, 2))
        axes.set_title("Top few targets")
        axes.get_figure().savefig(helpers.general._with_extra(helpers.general._with_extra(graph_filename, "targets"), "resample_{}".format(resample)), dpi=300)

        sampled_df = pred_df[cols]
        if resample:
            sampled_df = sampled_df.resample(resample).mean()
        axes = sampled_df.plot(figsize=(16, 9), ylim=(0, 2))
        axes.set_title("Top predictions")
        axes.get_figure().savefig(helpers.general._with_extra(helpers.sk.with_model_name(graph_filename, model, snake_case=True), "resample{}".format(resample)), dpi=300)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    dataset = load_split_data(args)

    elastic_net = sklearn.linear_model.ElasticNet(0.01)

    if args.clip:
        mlp = make_nn()[0]

        # load the clipper
        with io.open(args.clip, "rb") as json_in:
            clipper = helpers.sk.OutputClippedTransform.load(json_in)

        # wrap the model with the clipper
        elastic_net = helpers.sk.OutputTransformation(elastic_net, clipper)
        mlp = helpers.sk.OutputTransformation(mlp, clipper)
    else:
        mlp = make_nn()[0]

    cv_graph(dataset, elastic_net, args.base_output)
    cv_graph(dataset, mlp, args.base_output)

if __name__ == "__main__":
    sys.exit(main())
