from __future__ import unicode_literals
import sys
import argparse
from train_test import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resample", default="1H", help="Time interval to resample the training data")
    parser.add_argument("training_dir", help="Dir with the training CSV files")
    parser.add_argument("testing_dir", help="Dir with the testing files, including the empty prediction file")
    parser.add_argument("prediction_file", help="Destination for predictions")

    return parser.parse_args()


def predict_test_data(X_train, Y_train, scaler, testing_dir, testing_output):
    # retrain baseline model as a sanity check
    baseline_model = sklearn.dummy.DummyRegressor("mean").fit(X_train, Y_train)

    # retrain a model on the full data
    model = make_nn().fit(X_train, Y_train.values)

    test_data = load_data(testing_dir)
    X_test, Y_test = separate_output(test_data)
    X_test = scaler.transform(X_test)

    test_data[Y_train.columns] = model.predict(X_test)

    verify_predictions(X_test, baseline_model, model)

    # redo the index as unix timestamp
    test_data.index = test_data.index.astype(numpy.int64) / 10 ** 6
    test_data[Y_test.columns].to_csv(with_date(with_model_name(with_num_features(testing_output, X_train), model)), index_label="ut_ms")


def verify_predictions(X_test, baseline_model, model):
    baseline_predictions = baseline_model.predict(X_test)
    predictions = model.predict(X_test)

    deltas = numpy.abs(predictions - baseline_predictions) / numpy.abs(baseline_predictions)
    per_row = deltas.mean()

    unusual_rows = ~(per_row < 5)
    unusual_count = unusual_rows.sum()
    if unusual_count > 0:
        print "{:.1f}% ({:,} / {:,}) of rows have unusual predictions:".format(100. * unusual_count / predictions.shape[0], unusual_count, predictions.shape[0])

        unusual_inputs = X_test[unusual_rows].reshape(-1, X_test.shape[1])
        unusual_outputs = predictions[unusual_rows].reshape(-1, predictions.shape[1])

        for i in xrange(unusual_inputs.shape[0]):
            print "Input: ", unusual_inputs[i]
            print "Output: ", unusual_outputs[i]

    overall_delta = per_row.mean()
    print "Average percent change from baseline predictions: {:.2f}%".format(100. * overall_delta)

    assert overall_delta < 2


def with_num_features(filename, X):
    return filename.replace(".", ".{}_features.".format(X.shape[1]), 1)


def with_model_name(filename, model):
    return filename.replace(".", ".{}.".format(type(model).__name__), 1)


def with_date(filename):
    return filename.replace(".", ".{}.".format(datetime.datetime.now().strftime("%m_%d")), 1)


def main():
    args = parse_args()

    train_data = load_data(args.training_dir, resample_interval=args.resample, filter_null_power=True)

    X_train, Y_train = separate_output(train_data)

    scaler = make_scaler()

    X_train = scaler.fit_transform(X_train)

    predict_test_data(X_train, Y_train, scaler, args.testing_dir, args.prediction_file)


if __name__ == "__main__":
    sys.exit(main())