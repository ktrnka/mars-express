from __future__ import unicode_literals

"""
Run this to generate the preprocessed feature matrix so that you don't need to do it every time.
"""

import argparse
import logging
import sys

import loaders
from helpers.general import Timed, with_date, with_num_features, _with_extra


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--resample", default="1H", help="Time interval to resample the training data")
    parser.add_argument("--no-annotations", default=False, action="store_true", help="Disable date/num features annotations in the filename")
    parser.add_argument("training_dir", help="Dir with the training CSV files")
    parser.add_argument("output", help="Feature matrix output")

    return parser.parse_args()


@Timed
def main():
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG)

    for resample in args.resample.split(","):
        generate_csv(args.training_dir, resample, _with_extra(args.output, resample))


def generate_csv(training_dir, resample, output_filename):
    train_data = loaders.load_data(training_dir, resample_interval=resample, filter_null_power=True)
    X, _ = loaders.separate_output(train_data)
    train_data.to_csv(with_num_features(with_date(output_filename), X))


if __name__ == "__main__":
    sys.exit(main())
