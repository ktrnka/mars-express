from __future__ import unicode_literals
from __future__ import print_function
import sys
import argparse
import pandas


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file", help="Output predictions")
    parser.add_argument("prediction_file", nargs="+", help="Two or more prediction files to be averaged.")
    return parser.parse_args()


def main():
    args = parse_args()

    prediction_dfs = [pandas.read_csv(f, index_col=0) for f in args.prediction_file]

    mean_df = pandas.concat(prediction_dfs).groupby(level=0).mean()

    mean_df.to_csv(args.output_file, index_label="ut_ms")


if __name__ == "__main__":
    sys.exit(main())
