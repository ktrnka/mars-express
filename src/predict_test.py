from __future__ import print_function
from __future__ import unicode_literals

from helpers.general import with_num_features, with_model_name, with_date
from train_test import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resample", default="1H", help="Time interval to resample the training data")
    parser.add_argument("--model", default="nn", help="Model to generate predictions with")
    parser.add_argument("--graph", default=None, help="Graph to create of predictions for top few outputs compared to baseline")
    parser.add_argument("training_dir", help="Dir with the training CSV files")
    parser.add_argument("testing_dir", help="Dir with the testing files, including the empty prediction file")
    parser.add_argument("prediction_file", help="Destination for predictions")

    return parser.parse_args()

def get_model(model_name):
    model_name = model_name.lower()

    if model_name == "nn":
        return make_nn()
    elif model_name == "blr":
        return make_blr()
    else:
        raise ValueError("Unknown model abbreviation '{}'".format(model_name))


def graph_predictions(X_test, baseline_model, model, Y_train, output_file, test_index):
    important_columns = [c for c, _ in rate_columns(Y_train).most_common(5)]
    output_names = Y_train.columns

    Y_baseline = pandas.DataFrame(baseline_model.predict(X_test), columns=output_names, index=test_index)
    Y_baseline = Y_baseline[important_columns]
    Y_model = pandas.DataFrame(model.predict(X_test), columns=output_names, index=test_index)
    Y_model = Y_model[important_columns]

    axes = Y_baseline.resample("1D").mean().plot(figsize=(16, 9), ylim=(0, 2))
    axes.get_figure().savefig(with_model_name(output_file, baseline_model), dpi=300)

    axes = Y_model.resample("1D").mean().plot(figsize=(16, 9), ylim=(0, 2))
    axes.get_figure().savefig(with_model_name(output_file, model), dpi=300)


def rate_columns(data):
    # TODO: This is a candidate for shared lib
    """Rate columns by mean and stddev"""
    return collections.Counter({c: data[c].mean() + data[c].std() for c in data.columns})


def predict_test_data(X_train, Y_train, scaler, args):
    # retrain baseline model as a sanity check
    baseline_model = sklearn.dummy.DummyRegressor("mean").fit(X_train, Y_train)

    # retrain a model on the full data
    model = get_model(args.model).fit(X_train, Y_train.values)

    test_data = load_data(args.testing_dir)
    X_test, Y_test = separate_output(test_data)
    X_test = scaler.transform(X_test)

    test_data[Y_train.columns] = model.predict(X_test)

    verify_predictions(X_test, baseline_model, model)

    if args.graph:
        graph_predictions(X_test, baseline_model, model, Y_train, args.graph, test_data.index)

    # redo the index as unix timestamp
    test_data.index = test_data.index.astype(numpy.int64) / 10 ** 6
    test_data[Y_test.columns].to_csv(with_date(with_model_name(with_num_features(args.prediction_file, X_train), model)), index_label="ut_ms")


def verify_predictions(X_test, baseline_model, model):
    baseline_predictions = baseline_model.predict(X_test)
    predictions = model.predict(X_test)

    deltas = numpy.abs(predictions - baseline_predictions) / numpy.abs(baseline_predictions)
    per_row = deltas.mean()

    unusual_rows = ~(per_row < 5)
    unusual_count = unusual_rows.sum()
    if unusual_count > 0:
        print("{:.1f}% ({:,} / {:,}) of rows have unusual predictions:".format(100. * unusual_count / predictions.shape[0], unusual_count, predictions.shape[0]))

        unusual_inputs = X_test[unusual_rows].reshape(-1, X_test.shape[1])
        unusual_outputs = predictions[unusual_rows].reshape(-1, predictions.shape[1])

        for i in xrange(unusual_inputs.shape[0]):
            print(("Input: ", unusual_inputs[i]))
            print(("Output: ", unusual_outputs[i]))

    overall_delta = per_row.mean()
    print("Average percent change from baseline predictions: {:.2f}%".format(100. * overall_delta))

    assert overall_delta < 2


def main():
    args = parse_args()

    train_data = load_data(args.training_dir, resample_interval=args.resample, filter_null_power=True)

    X_train, Y_train = separate_output(train_data)

    scaler = make_scaler()

    X_train = scaler.fit_transform(X_train)

    predict_test_data(X_train, Y_train, scaler, args)


if __name__ == "__main__":
    sys.exit(main())