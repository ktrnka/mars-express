from __future__ import unicode_literals
from __future__ import print_function

import collections
import sys
import argparse

import operator
import pandas
"""
Diff two pandas CSV files and spit out some info.
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", nargs="+")
    return parser.parse_args()


def diff_dataframes(*dataframes):
    if not all(len(dataframes[0]) == len(df) for df in dataframes[1:]):
        print("Dataframes have different lengths:", [len(df) for df in dataframes])
    else:
        print("All dataframes have the same number of rows")
    # features
    feature_list = [set(df.columns) for df in dataframes]

    shared_features = reduce(operator.__and__, feature_list[1:], feature_list[0])

    print("Shared features:", ", ".join(sorted(shared_features)))

    for i, feature_set in enumerate(feature_list):
        print("In file {} only: {}".format(i, sorted(shared_features - feature_set)))

    for i in range(1, len(dataframes)):
        print("Comparing file {} to 0".format(i))
        feature_diffs = dict()
        for feature in shared_features:
            feature_diffs[feature] = dataframes[i][feature].mean() - dataframes[0][feature].mean()

        diff_features = [f for f in feature_diffs.keys() if feature_diffs[f] != 0]
        diff_features = sorted(diff_features, key=lambda f: abs(feature_diffs[f]), reverse=True)

        if not diff_features:
            print("\tNo features with difference of means")
        else:
            for feature in diff_features:
                print("\t{}: {} diff of means".format(feature, diff_features[feature]))



def main():
    args = parse_args()

    diff_dataframes(*[pandas.read_csv(f, index_col=0) for f in args.csv_files])


if __name__ == "__main__":
    sys.exit(main())