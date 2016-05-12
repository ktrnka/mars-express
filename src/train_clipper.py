from __future__ import unicode_literals
from __future__ import print_function

import logging
import sys
import argparse

import io

import train_test
import helpers.sk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("training_dir",
                        help="Dir with the training CSV files or joined CSV file with the complete feature matrix")
    parser.add_argument("clipper_file", help="Clipper saved weights (Json)")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    unsampled_outputs = train_test.load_series(train_test.find_files(args.training_dir, "power")).dropna()
    clipper = helpers.sk.OutputClippedTransform.from_data(unsampled_outputs.values)

    with io.open(args.clipper_file, "wb") as json_out:
        clipper.save(json_out)


if __name__ == "__main__":
    sys.exit(main())
