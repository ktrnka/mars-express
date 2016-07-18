from __future__ import print_function
from __future__ import unicode_literals

import argparse
import collections
import logging
import os
import sys
from operator import itemgetter
from pprint import pprint

import numpy
import pandas
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
from helpers.debug import verify_data, compute_cross_validation_fairness
from helpers.features import add_lag_feature, add_transformation_feature, get_event_series, TimeRange, roll
from helpers.sk import rms_error


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--debug", default=False, action="store_true", help="Debug logging")

    parser.add_argument("--time-steps", default=4, type=int, help="Number of time steps for recurrent/etc models")
    parser.add_argument("--verify", default=False, action="store_true", help="Run verifications on the input data for outliers and such")

    parser.add_argument("--feature-pairs", default=False, action="store_true", help="Try out pairs of features")
    parser.add_argument("--resample", default="1H", help="Time interval to resample the training data")
    parser.add_argument("--extra-analysis", default=False, action="store_true", help="Extra analysis on the data")
    parser.add_argument("--analyse-feature-importance", default=False, action="store_true", help="Analyse feature importance and print them out for some models")
    parser.add_argument("--analyse-hyperparameters", default=False, action="store_true", help="Analyse hyperparameters and print them out for some models")

    parser.add_argument("training_dir", help="Dir with the training CSV files or joined CSV file with the complete feature matrix")
    return parser.parse_args()


def parse_dates(column):
    """Date parser for pandas.read_csv"""
    return pandas.to_datetime(column.astype(int), unit="ms")


def find_files(path, substring):
    files = os.listdir(path)
    return sorted(os.path.join(path, f) for f in files if substring in f)


def get_evtf_ranges(event_data, event_prefix):
    """Get time ranges between event_prefix + _START and event_prefix + _END"""
    current_start = None
    event_ranges = []
    for date, row in event_data[event_data.description.str.startswith(event_prefix)].iterrows():
        if row["description"].endswith("_START"):
            current_start = date
        elif current_start:
            assert row["description"].endswith("_END")
            event_ranges.append(TimeRange(current_start, date))
            current_start = None
    return event_ranges


def get_dmop_subsystem(dmop_data, include_command=False):
    """Extract the subsystem from each record of the dmop data"""
    dmop_subsys = dmop_data.subsystem.str.extract(r"A(?P<subsystem>\w{3})(?P<command>.*)", expand=False)
    dmop_subsys_mapo = dmop_data.subsystem.str.extract(r"(?P<subsystem>.+)\..+", expand=False)

    dmop_subsys.subsystem.fillna(dmop_subsys_mapo, inplace=True)
    dmop_subsys.subsystem.fillna(dmop_data.subsystem, inplace=True)

    if include_command:
        return dmop_subsys.subsystem.str.cat(dmop_subsys.command, sep="_")
    else:
        return dmop_subsys.subsystem


def get_communication_latency(earthmars_km_series):
    """Get the one-way latency for Earth-Mars communication as a Series"""
    return pandas.to_timedelta(earthmars_km_series/ 2.998e+5, unit="s")


def adjust_for_latency(data, latency_series, earth_sent=True):
    """Adjust the datetimeindex of data for Earth-Mars communication latency. If the data is sent from earth, add
    the latency. If the data is received on earth, subtract it to convert to spacecraft time."""
    offsets = latency_series.reindex(index=data.index, method="nearest")

    if earth_sent:
        data.index = data.index + offsets
    else:
        data.index = data.index - offsets


def adjust_for_latency_ftl(data, latency_series):
    """Adjust the datetimeindex of data for Earth-Mars communication latency. If the data is sent from earth, add
    the latency. If the data is received on earth, subtract it to convert to spacecraft time."""
    offsets = latency_series.reindex(index=data.index, method="nearest")

    data.index = data.index + offsets
    data["ute_ms"] = data["ute_ms"] + offsets


def get_dmop_ranges(dmop_subsystem, subsystem_name, hours_impact=1.):
    time_offset = pandas.Timedelta(hours=hours_impact)
    for t in dmop_subsystem[dmop_subsystem == subsystem_name].index:
        yield TimeRange(t, t + time_offset)


def get_evtf_altitude(event_data, index=None):
    """Read the ascend/descend events and compute the current altitude at all points"""
    desc = event_data.description.str.extract(r"(\d+)_KM_DESCEND", expand=False).fillna(0).astype(numpy.int16)
    asc = event_data.description.str.extract(r"(\d+)_KM_ASCEND", expand=False).fillna(0).astype(numpy.int16)

    alt_delta = asc - desc
    alt = alt_delta.cumsum()
    alt -= alt.min()

    if index is not None:
        alt = alt.resample("1H").mean().interpolate().reindex(index, method="nearest")

    return alt


def get_ftl_periods(ftl_slice):
    """Get time ranges for FTL data (first two columns are start and end time so it's simple)"""
    for row in ftl_slice.itertuples():
        yield TimeRange(row[0], row[1])


def load_series(files, add_file_number=False, resample_interval=None, roll_mean_window=None, date_cols=True):
    data = [pandas.read_csv(f, parse_dates=date_cols, date_parser=parse_dates, index_col=0) for f in files]

    if resample_interval:
        data = [d.resample(resample_interval).mean() for d in data]

    if roll_mean_window:
        data = [d.rolling(window=roll_mean_window, min_periods=1).mean() for d in data]

    if add_file_number:
        for i, year_data in enumerate(data):
            year_data["file_number"] = i

    data = pandas.concat(data)
    assert isinstance(data, pandas.DataFrame)

    return data


def add_integrated_distance_feature(dataframe, feature, point, num_periods):
    """Derive a feature with a simple function like log, sqrt, etc"""
    assert isinstance(dataframe, pandas.DataFrame)
    new_name = "{}_dist_from_{}_rolling{}".format(feature, point, num_periods)

    transformed = (dataframe[feature] - point) ** -2
    transformed = transformed.rolling(num_periods).mean().fillna(method="bfill")

    dataframe[new_name] = transformed


def centered_ewma(series, num_spans):
    forward = series.ewm(span=num_spans).mean()
    backward = series[::-1].ewm(span=num_spans).mean()[::-1]

    means = numpy.vstack([forward.values, backward.values]).mean(axis=0)

    return pandas.Series(index=forward.index, data=means, name=series.name)


def add_ewma(dataframe, feature, num_spans=24*7, drop=False):
    assert isinstance(dataframe, pandas.DataFrame)
    new_name = "{}_ewma{}".format(feature, num_spans)
    dataframe[new_name] = centered_ewma(dataframe[feature], num_spans=num_spans)

    if drop:
        dataframe.drop(feature, axis=1, inplace=True)


def time_since_last_event(event_data, index):
    """Make a Series with the specified index that tracks time since the last event, backfilled with zero"""
    event_dates = pandas.Series(index=event_data.index, data=event_data.index, name="date")
    event_dates = event_dates.reindex(index, method="ffill")
    deltas = event_dates.index - event_dates
    return deltas.fillna(0).dt.total_seconds()


def get_signal_level(filtered_event_data, index):
    """Merge AOS/LOS signals into a relatively simplistic rolling sum of all AOS and LOS"""
    # earth_evtf = event_data[event_data.description.str.contains("RTLT")]
    signals = filtered_event_data.description.str.contains("AOS").astype("int") - filtered_event_data.description.str.contains("LOS").astype(int)
    signals_sum = signals.cumsum()

    # there's long total loss of signal during conjunctions so this helps to compensate
    signals_mins = signals_sum.rolling(100, min_periods=1).min()
    signals_max = signals_sum.rolling(100, min_periods=1).max().astype(float)
    signals_sum = (signals_sum - signals_mins) / (signals_max - signals_mins)

    # drop index duplicates after the normalization
    signals_sum = signals_sum.groupby(level=0).first()

    return signals_sum.reindex(index, method="ffill").fillna(method="backfill")


def hourly_event_count(event_data, index):
    """Make a Series with the specified index that has the hourly count of events in event_data"""
    if len(event_data) == 0:
        return pandas.Series(index=index, data=0)

    event_counts = pandas.Series(index=event_data.index, data=event_data.index, name="date")

    # TODO - needs to be next 1h to match the power resample
    event_counts = event_counts.resample("5Min").count().rolling(12).sum().bfill()
    return event_counts.reindex(index, method="nearest")

def hourly_event_count_fixed(event_data, index):
    """Make a Series with the specified index that has the hourly count of events in event_data"""
    if len(event_data) == 0:
        return pandas.Series(index=index, data=0)

    event_counts = pandas.Series(index=event_data.index, data=event_data.index, name="date")

    event_counts = event_counts.resample("5Min").count()[::-1].rolling(12).sum().bfill()[::-1]
    return event_counts.reindex(index, method="nearest")


def parse_cut_feature(range_feature_name):
    """Parse a feature like sz__(95.455, 101.35] into sz, 95.455, 101.35"""
    base_name, ranges = range_feature_name.split("__")
    lower, upper = ranges[1:-1].split(", ")

    return base_name, float(lower), float(upper)


def auto_log(data, columns):
    logger = helpers.general.get_function_logger()
    X = sklearn.preprocessing.RobustScaler().fit_transform(data[columns])

    # find rows with values over 10x IQR from median
    X_deviants = numpy.abs(X) > 10
    column_deviants = X_deviants.mean(axis=0)

    changed_cols = []
    for column, deviation in zip(columns, column_deviants):
        if deviation > 0.01:
            logger.info("Auto log on %s with %.1f%% deviant values", column, 100. * deviation)
            add_transformation_feature(data, column, "log", drop=True)
            changed_cols.append(column)

    logger.info("Changed columns: %s", changed_cols)

def make_label(hours):
    base = ""
    if hours <= 0:
        base = "next"
        hours = -hours

    if hours >= 24:
        base += "{}d".format(hours / 24)
    else:
        base += "{}h".format(hours)

    return base

@helpers.general.Timed
def load_data(data_dir, resample_interval=None, filter_null_power=False, derived_features=True):
    logger = helpers.general.get_function_logger()

    if not os.path.isdir(data_dir):
        return pandas.read_csv(data_dir, index_col=0, parse_dates=True)

    # load the base power data
    data = load_series(find_files(data_dir, "power"), add_file_number=True, resample_interval=resample_interval)

    event_sampling_index = pandas.DatetimeIndex(freq="5Min", start=data.index.min(), end=data.index.max())
    event_sampled_df = pandas.DataFrame(index=event_sampling_index)

    ### LTDATA ###
    longterm_data = load_series(find_files(data_dir, "ltdata"))

    # as far as I can tell this doesn't make a difference but it makes me feel better
    longterm_data = longterm_data.resample("1H").mean().interpolate().fillna(method="backfill")

    one_way_latency = get_communication_latency(longterm_data.earthmars_km)

    # time-lagged version
    add_lag_feature(longterm_data, "eclipseduration_min", 2 * 24, "2d", data_type=numpy.int64)
    add_lag_feature(longterm_data, "eclipseduration_min", 5 * 24, "5d", data_type=numpy.int64)
    add_lag_feature(longterm_data, "eclipseduration_min", 64 * 24, "64d")
    add_lag_feature(longterm_data, "sunmars_km", 24, "1d", drop=True)
    add_lag_feature(longterm_data, "sunmarsearthangle_deg", 24 * 64, "64d")

    ### FTL ###
    ftl_data = load_series(find_files(data_dir, "ftl"), date_cols=["utb_ms", "ute_ms"])

    event_sampled_df["flagcomms"] = get_event_series(event_sampling_index, get_ftl_periods(ftl_data[ftl_data.flagcomms]))

    # TODO - needs to be next 1h to match the power resample
    add_lag_feature(event_sampled_df, "flagcomms", 12, "1h")
    add_lag_feature(event_sampled_df, "flagcomms", 24, "2h")
    event_sampled_df.drop("flagcomms", axis=1, inplace=True)

    # select columns or take preselected ones
    for ftl_type in ["SLEW", "EARTH", "INERTIAL", "D4PNPO", "MAINTENANCE", "NADIR", "WARMUP", "ACROSS_TRACK", "RADIO_SCIENCE"]:
        dest_name = "FTL_" + ftl_type
        event_sampled_df[dest_name] = get_event_series(event_sampled_df.index, get_ftl_periods(ftl_data[ftl_data["type"] == ftl_type]))

        # TODO - needs to be next 1h to match the power resample
        add_lag_feature(event_sampled_df, dest_name, 12, "1h")
        add_lag_feature(event_sampled_df, dest_name, 24, "2h")

        if ftl_type == "MAINTENANCE" or ftl_type == "EARTH" or ftl_type == "SLEW":
            add_lag_feature(event_sampled_df, dest_name, -2 * 12, make_label(-2))
        event_sampled_df.drop(dest_name, axis=1, inplace=True)

    ### EVTF ###
    event_data = load_series(find_files(data_dir, "evtf"))

    for event_name in ["MAR_UMBRA", "MRB_/_RANGE_06000KM", "MSL_/_RANGE_06000KM"]:
        dest_name = "EVTF_IN_" + event_name
        event_sampled_df[dest_name] = get_event_series(event_sampling_index, get_evtf_ranges(event_data, event_name))

        # TODO - needs to be next 1h to match the power resample
        add_lag_feature(event_sampled_df, dest_name, 12, "1h")

        if event_name == "MAR_UMBRA":
            add_lag_feature(event_sampled_df, dest_name, 8 * 12, "8h")
            add_lag_feature(event_sampled_df, dest_name, -8 * 12, "next8h")
        event_sampled_df.drop(dest_name, axis=1, inplace=True)

    # event_sampled_df["EVTF_EARTH_LOS"] = get_earth_los(event_data, event_sampled_df.index).rolling(12, min_periods=0).mean()

    event_sampled_df["EVTF_TIME_MRB_AOS_10"] = time_since_last_event(event_data[event_data.description == "MRB_AOS_10"], event_sampled_df.index)
    event_sampled_df["EVTF_TIME_MRB_AOS_00"] = time_since_last_event(event_data[event_data.description == "MRB_AOS_00"], event_sampled_df.index)
    event_sampled_df["EVTF_TIME_MSL_AOS_10"] = time_since_last_event(event_data[event_data.description == "MSL_AOS_10"], event_sampled_df.index)

    altitude_series = get_evtf_altitude(event_data, index=data.index)
    event_data.drop(["description"], axis=1, inplace=True)
    event_data["EVTF_event_counts"] = 1

    # TODO - needs to be next 1h to match the power resample
    event_data = event_data.resample("5Min").count().rolling(12).sum().fillna(method="bfill").reindex(data.index, method="nearest")
    event_data["EVTF_altitude"] = altitude_series

    # TODO - needs to be next 1h to match the power resample
    add_lag_feature(event_data, "EVTF_event_counts", 2, "2h", data_type=numpy.int64)
    add_lag_feature(event_data, "EVTF_event_counts", 5, "5h", data_type=numpy.int64)

    ### DMOP ###
    dmop_data = load_series(find_files(data_dir, "dmop"))

    # TODO - this may be incorrect and might've only been a slight help because of the bug with resampling
    adjust_for_latency(dmop_data, one_way_latency)

    dmop_subsystems = get_dmop_subsystem(dmop_data, include_command=False)


    # these subsystems were found partly by trial and error
    # for subsys in dmop_subsystems.value_counts().sort_values(ascending=False).index[:100]:
    for subsys in "AAA PSF ACF MMM TTT SSS HHH OOO MAPO MPER MOCE MOCS PENS PENE TMB VVV SXX".split():
        dest_name = "DMOP_{}_event_count".format(subsys)
        event_sampled_df[dest_name] = hourly_event_count(dmop_subsystems[dmop_subsystems == subsys], event_sampled_df.index)

    subsystem_windows = {
        "OOO": [-4, 12],
        "PSF": [-4],
        "VVV": [-4],
        "SXX": [-12],
        "PENS": [-4],
        "MOCE": [-4],
    }
    for subsys, windows in subsystem_windows.items():
        dest_name = "DMOP_{}_event_count".format(subsys)
        for hours in windows:
            add_lag_feature(event_sampled_df, dest_name, hours * 12, make_label(hours))

    subsystem_windows = {
        "OOO_F77A0": 4,
        "OOO_F68A0": 4,
        # "ACF_M06A": 4,
        # "ACF_E05A": 24 * 16,
        "TTT_305O": 24 * 4,
        "SXX_307A": 24 * 4,
        "SXX_303A": 4,
        "SXX_382C": 24 * 4,
        # "PSF_31B1": 4,
        "SSS_F53A0": 24 * 64,
    }
    # subsystems with the command included just for a few
    dmop_subsystems = get_dmop_subsystem(dmop_data, include_command=True)
    for subsys, hours in subsystem_windows.items():
        dest_name = "DMOP_{}_event_count".format(subsys)
        event_sampled_df[dest_name] = hourly_event_count(dmop_subsystems[dmop_subsystems == subsys], event_sampled_df.index)
        add_lag_feature(event_sampled_df, dest_name, hours * 12, make_label(hours), drop=True)

    dmop_data.drop(["subsystem"], axis=1, inplace=True)
    dmop_data["DMOP_event_counts"] = 1

    # TODO - needs to be next 1h to match the power resample
    dmop_data = dmop_data.resample("5Min").count().rolling(12).sum().fillna(method="bfill").reindex(data.index, method="nearest")
    add_lag_feature(dmop_data, "DMOP_event_counts", 2, "2h", data_type=numpy.int64)
    add_lag_feature(dmop_data, "DMOP_event_counts", 5, "5h", data_type=numpy.int64)

    ### SAAF ###
    saaf_data = load_series(find_files(data_dir, "saaf"))

    # try a totally different style
    # saaf_data = saaf_data.resample("15Min").mean().interpolate().rolling(4).mean().fillna(method="bfill").reindex(data.index, method="nearest")
    saaf_data = saaf_data.resample("2Min").mean().interpolate()
    saaf_periods = 30

    saaf_quartiles = []
    # for col in ["sx", "sy", "sz", "sa"]:
    #     quartile_indicator_df = pandas.get_dummies(pandas.qcut(saaf_data[col], 10), col + "_")
    #     quartile_hist_df = quartile_indicator_df.rolling(saaf_periods, min_periods=1).mean()
    #     saaf_quartiles.append(quartile_hist_df)

    feature_weights = [(u'sz__(85.86, 90.0625]', 0.029773711557758643),
                       (u'sa__(1.53, 5.192]', 0.018961685024380826),
                       (u'sy__(89.77, 89.94]', 0.017065361875026698),
                       (u'sx__(2.355, 5.298]', 0.016407419574092953),
                       (u'sx__(32.557, 44.945]', 0.015424066115462104),
                       (u'sz__(101.35, 107.00167]', 0.012587397977008816),
                       (u'sz__(112.47, 117.565]', 0.010126162391495826),
                       (u'sz__(107.00167, 112.47]', 0.0099830744754197034),
                       (u'sz__(117.565, 121.039]', 0.0081198807115996485),
                       (u'sa__(5.192, 18.28]', 0.0079614117402690421),
                       (u'sy__(89.94, 90]', 0.0064161463399279106),
                       (u'sz__(90.0625, 95.455]', 0.0060623602567580299),
                       (u'sa__(0.739, 1.53]', 0.0050941311789206674),
                       (u'sa__(0.198, 0.31]', 0.00064943741967410915)]

    for feature, weight in feature_weights:
        if feature.startswith("s") and "__" in feature:
            base, lower, upper = parse_cut_feature(feature)

            indicators = (saaf_data[base] > lower) & (saaf_data[base] <= upper)

            # TODO - needs to be next 1h to match the power resample
            rolling_count = indicators.rolling(saaf_periods, min_periods=1).mean()
            rolling_count.rename(feature, inplace=True)
            saaf_quartiles.append(rolling_count)

    saaf_quartile_df = pandas.concat(saaf_quartiles, axis=1).reindex(data.index, method="nearest")

    # 3 delays because 60 min / 5 min = 12
    # for col in ["sx", "sy", "sz", "sa"]:
    #     for delay in range(1, 30):
    #         saaf_data["{}_prev{}".format(col, delay)] = saaf_data[col].shift(delay)

    # convert to simple rolling mean
    # TODO - needs to be next 1h to match the power resample
    saaf_data = saaf_data.rolling(saaf_periods).mean().fillna(method="bfill")

    # SAAF rolling stddev, took top 2 from ElasticNet
    # TODO: This is fine cause it's a long range window
    for num_days in [1, 8]:
        saaf_data["SAAF_stddev_{}d".format(num_days)] = saaf_data[["sx", "sy", "sz", "sa"]].rolling(num_days * 24 * saaf_periods).std().fillna(method="bfill").sum(axis=1)
    saaf_data = saaf_data.reindex(data.index, method="nearest").fillna(method="bfill")

    longterm_data = longterm_data.reindex(data.index, method="nearest")

    data = pandas.concat([data, saaf_data, longterm_data, dmop_data, event_data, event_sampled_df.reindex(data.index, method="nearest"), saaf_quartile_df], axis=1)
    assert isinstance(data, pandas.DataFrame)

    if filter_null_power:
        previous_size = data.shape[0]
        data = data[data.NPWD2532.notnull()]
        if data.shape[0] < previous_size:
            logger.info("Reduced data from {:,} rows to {:,}".format(previous_size, data.shape[0]))

    data["days_in_space"] = (data.index - pandas.datetime(year=2003, month=6, day=2)).days

    if derived_features:
        for col in [c for c in data.columns if "EVTF_IN_MRB" in c]:
            add_transformation_feature(data, col, "gradient")
        add_transformation_feature(data, "FTL_EARTH_rolling_1h", "gradient")
        add_transformation_feature(data, "DMOP_event_counts", "log", drop=True)
        add_transformation_feature(data, "DMOP_event_counts_rolling_2h", "gradient", drop=True)
        add_transformation_feature(data, "occultationduration_min", "log", drop=True)

        add_transformation_feature(data, "sa", "log", drop=True)
        add_transformation_feature(data, "sy", "log", drop=True)

        # # various crazy rolling features
        add_lag_feature(data, "EVTF_IN_MAR_UMBRA_rolling_1h", 50, "50")
        add_lag_feature(data, "EVTF_IN_MRB_/_RANGE_06000KM_rolling_1h", 1600, "1600")
        add_lag_feature(data, "EVTF_event_counts_rolling_5h", 50, "50")
        # add_lag_feature(data, "FTL_ACROSS_TRACK_rolling_1h", 200, "200")
        add_lag_feature(data, "FTL_NADIR_rolling_1h", 400, "400")

    # data.drop([u'EVTF_event_counts', u'days_in_space', u'DMOP_PSF_event_count_rolling_next4h', u'SAAF_stddev_1d', u'DMOP_MAPO_event_count', u'DMOP_OOO_F77A0_event_count_rolling_4h', u'DMOP_MOCE_event_count', u'DMOP_event_counts_rolling_2h_gradient', u'DMOP_AAA_event_count', u'sz__(117.565, 121.039]', u'FTL_INERTIAL_rolling_1h', u'sa__(5.192, 18.28]', u'FTL_EARTH_rolling_1h_gradient', u'sa_log', u'eclipseduration_min_rolling_5d', u'sx__(32.557, 44.945]', u'EVTF_TIME_MSL_AOS_10', u'FTL_WARMUP_rolling_1h', u'sa__(1.53, 5.192]', u'sz__(85.86, 90.0625]', u'DMOP_SXX_event_count', u'DMOP_HHH_event_count', u'sz__(112.47, 117.565]', u'DMOP_event_counts_log', u'FTL_SLEW_rolling_1h', u'sy__(89.77, 89.94]', u'DMOP_PSF_event_count', u'DMOP_PENE_event_count', u'DMOP_MOCE_event_count_rolling_next4h', u'EVTF_TIME_MRB_AOS_00', u'EVTF_IN_MAR_UMBRA_rolling_next8h', u'sz__(107.00167, 112.47]', u'FTL_EARTH_rolling_next2h', u'sa__(0.739, 1.53]', u'sy__(89.94, 90]', u'EVTF_TIME_MRB_AOS_10', u'DMOP_MPER_event_count', u'DMOP_SSS_event_count', u'sz__(101.35, 107.00167]', u'sx__(2.355, 5.298]', u'DMOP_event_counts_rolling_5h', u'sz__(90.0625, 95.455]', u'sy_log', u'DMOP_TTT_event_count', u'FTL_ACROSS_TRACK_rolling_1h', u'FTL_INERTIAL_rolling_2h', u'FTL_SLEW_rolling_2h', 'eclipseduration_min', u'sa__(0.198, 0.31]', u'EVTF_IN_MAR_UMBRA_rolling_1h_rolling_50', u'eclipseduration_min_rolling_2d', u'FTL_MAINTENANCE_rolling_1h', u'FTL_NADIR_rolling_1h', u'FTL_ACROSS_TRACK_rolling_2h', u'FTL_RADIO_SCIENCE_rolling_2h'], axis=1, inplace=True)

    # data.drop("earthmars_km", axis=1, inplace=True)

    logger.info("DataFrame shape %s", data.shape)
    return data


@helpers.general.Timed
def load_data_fixed(data_dir, resample_interval=None, filter_null_power=False, derived_features=True):
    logger = helpers.general.get_function_logger()

    if not os.path.isdir(data_dir):
        return pandas.read_csv(data_dir, index_col=0, parse_dates=True)

    # load the base power data
    data = load_series(find_files(data_dir, "power"), add_file_number=True, resample_interval=resample_interval)

    event_sampling_index = pandas.DatetimeIndex(freq="5Min", start=data.index.min(), end=data.index.max())
    event_sampled_df = pandas.DataFrame(index=event_sampling_index)

    ### LTDATA ###
    longterm_data = load_series(find_files(data_dir, "ltdata"))

    # as far as I can tell this doesn't make a difference but it makes me feel better
    longterm_data = longterm_data.resample("1H").mean().interpolate().fillna(method="backfill")

    one_way_latency = get_communication_latency(longterm_data.earthmars_km)

    # time-lagged version
    add_lag_feature(longterm_data, "eclipseduration_min", 2 * 24, "2d", data_type=numpy.int64)
    add_lag_feature(longterm_data, "eclipseduration_min", 5 * 24, "5d", data_type=numpy.int64)
    add_lag_feature(longterm_data, "eclipseduration_min", 64 * 24, "64d")
    add_lag_feature(longterm_data, "sunmars_km", 24, "1d", drop=True)
    add_lag_feature(longterm_data, "sunmarsearthangle_deg", 24 * 64, "64d")

    ### FTL ###
    ftl_data = load_series(find_files(data_dir, "ftl"), date_cols=["utb_ms", "ute_ms"])

    event_sampled_df["flagcomms"] = get_event_series(event_sampling_index, get_ftl_periods(ftl_data[ftl_data.flagcomms]))
    add_lag_feature(event_sampled_df, "flagcomms", -12, "next1h")
    add_lag_feature(event_sampled_df, "flagcomms", 24, "2h")
    event_sampled_df.drop("flagcomms", axis=1, inplace=True)

    # select columns or take preselected ones
    for ftl_type in ["SLEW", "EARTH", "INERTIAL", "D4PNPO", "MAINTENANCE", "NADIR", "WARMUP", "ACROSS_TRACK", "RADIO_SCIENCE"]:
        dest_name = "FTL_" + ftl_type
        event_sampled_df[dest_name] = get_event_series(event_sampled_df.index, get_ftl_periods(ftl_data[ftl_data["type"] == ftl_type]))

        add_lag_feature(event_sampled_df, dest_name, -12, "next1h")
        add_lag_feature(event_sampled_df, dest_name, 24, "2h")

        if ftl_type == "MAINTENANCE" or ftl_type == "EARTH" or ftl_type == "SLEW":
            add_lag_feature(event_sampled_df, dest_name, -2 * 12, make_label(-2))
        event_sampled_df.drop(dest_name, axis=1, inplace=True)

    ### EVTF ###
    event_data = load_series(find_files(data_dir, "evtf"))

    for event_name in ["MAR_UMBRA", "MRB_/_RANGE_06000KM", "MSL_/_RANGE_06000KM"]:
        dest_name = "EVTF_IN_" + event_name
        event_sampled_df[dest_name] = get_event_series(event_sampling_index, get_evtf_ranges(event_data, event_name))

        add_lag_feature(event_sampled_df, dest_name, -12, "next1h")

        if event_name == "MAR_UMBRA":
            add_lag_feature(event_sampled_df, dest_name, 8 * 12, "8h")
            add_lag_feature(event_sampled_df, dest_name, -8 * 12, "next8h")
        event_sampled_df.drop(dest_name, axis=1, inplace=True)

    # event_sampled_df["EVTF_EARTH_LOS"] = get_earth_los(event_data, event_sampled_df.index).rolling(12, min_periods=0).mean()

    event_sampled_df["EVTF_TIME_MRB_AOS_10"] = time_since_last_event(event_data[event_data.description == "MRB_AOS_10"], event_sampled_df.index)
    event_sampled_df["EVTF_TIME_MRB_AOS_00"] = time_since_last_event(event_data[event_data.description == "MRB_AOS_00"], event_sampled_df.index)
    event_sampled_df["EVTF_TIME_MSL_AOS_10"] = time_since_last_event(event_data[event_data.description == "MSL_AOS_10"], event_sampled_df.index)

    altitude_series = get_evtf_altitude(event_data, index=data.index)
    event_data.drop(["description"], axis=1, inplace=True)
    event_data["EVTF_event_counts"] = 1

    event_data = event_data.resample("5Min").count()
    event_data = roll(event_data, -12, "sum").reindex(data.index, method="nearest")
    event_data["EVTF_altitude"] = altitude_series

    add_lag_feature(event_data, "EVTF_event_counts", 2, "2h", data_type=numpy.int64)
    add_lag_feature(event_data, "EVTF_event_counts", 5, "5h", data_type=numpy.int64)

    ### DMOP ###
    dmop_data = load_series(find_files(data_dir, "dmop"))

    # TODO - this may be incorrect and might've only been a slight help because of the bug with resampling
    adjust_for_latency(dmop_data, one_way_latency)

    dmop_subsystems = get_dmop_subsystem(dmop_data, include_command=False)

    # these subsystems were found partly by trial and error
    # for subsys in dmop_subsystems.value_counts().sort_values(ascending=False).index[:100]:
    for subsys in "AAA PSF ACF MMM TTT SSS HHH OOO MAPO MPER MOCE MOCS PENS PENE TMB VVV SXX".split():
        dest_name = "DMOP_{}_event_count".format(subsys)
        event_sampled_df[dest_name] = hourly_event_count_fixed(dmop_subsystems[dmop_subsystems == subsys], event_sampled_df.index)

    subsystem_windows = {
        "OOO": [-4, 12],
        "PSF": [-4],
        "VVV": [-4],
        "SXX": [-12],
        "PENS": [-4],
        "MOCE": [-4],
    }
    for subsys, windows in subsystem_windows.items():
        dest_name = "DMOP_{}_event_count".format(subsys)
        for hours in windows:
            add_lag_feature(event_sampled_df, dest_name, hours * 12, make_label(hours))

    subsystem_windows = {
        "OOO_F77A0": 4,
        "OOO_F68A0": 4,
        # "ACF_M06A": 4,
        # "ACF_E05A": 24 * 16,
        "TTT_305O": 24 * 4,
        "SXX_307A": 24 * 4,
        "SXX_303A": 4,
        "SXX_382C": 24 * 4,
        # "PSF_31B1": 4,
        "SSS_F53A0": 24 * 64,
    }
    # subsystems with the command included just for a few
    dmop_subsystems = get_dmop_subsystem(dmop_data, include_command=True)
    for subsys, hours in subsystem_windows.items():
        dest_name = "DMOP_{}_event_count".format(subsys)
        event_sampled_df[dest_name] = hourly_event_count_fixed(dmop_subsystems[dmop_subsystems == subsys], event_sampled_df.index)
        add_lag_feature(event_sampled_df, dest_name, hours * 12, make_label(hours), drop=True)

    dmop_data.drop(["subsystem"], axis=1, inplace=True)
    dmop_data["DMOP_event_counts"] = 1

    dmop_data = dmop_data.resample("5Min").count()
    dmop_data = roll(dmop_data, -12, "sum").reindex(data.index, method="nearest")

    add_lag_feature(dmop_data, "DMOP_event_counts", 2, "2h", data_type=numpy.int64)
    add_lag_feature(dmop_data, "DMOP_event_counts", 5, "5h", data_type=numpy.int64)

    ### SAAF ###
    saaf_data = load_series(find_files(data_dir, "saaf"))

    saaf_data = saaf_data.resample("2Min").mean().interpolate()
    saaf_periods = 30

    saaf_quartiles = []
    # for col in ["sx", "sy", "sz", "sa"]:
    #     quartile_indicator_df = pandas.get_dummies(pandas.qcut(saaf_data[col], 10), col + "_")
    #     quartile_hist_df = quartile_indicator_df.rolling(saaf_periods, min_periods=1).mean()
    #     saaf_quartiles.append(quartile_hist_df)

    feature_weights = [(u'sz__(85.86, 90.0625]', 0.029773711557758643),
                       (u'sa__(1.53, 5.192]', 0.018961685024380826),
                       (u'sy__(89.77, 89.94]', 0.017065361875026698),
                       (u'sx__(2.355, 5.298]', 0.016407419574092953),
                       (u'sx__(32.557, 44.945]', 0.015424066115462104),
                       (u'sz__(101.35, 107.00167]', 0.012587397977008816),
                       (u'sz__(112.47, 117.565]', 0.010126162391495826),
                       (u'sz__(107.00167, 112.47]', 0.0099830744754197034),
                       (u'sz__(117.565, 121.039]', 0.0081198807115996485),
                       (u'sa__(5.192, 18.28]', 0.0079614117402690421),
                       (u'sy__(89.94, 90]', 0.0064161463399279106),
                       (u'sz__(90.0625, 95.455]', 0.0060623602567580299),
                       (u'sa__(0.739, 1.53]', 0.0050941311789206674),
                       (u'sa__(0.198, 0.31]', 0.00064943741967410915)]

    for feature, weight in feature_weights:
        if feature.startswith("s") and "__" in feature:
            base, lower, upper = parse_cut_feature(feature)

            indicators = (saaf_data[base] > lower) & (saaf_data[base] <= upper)

            rolling_count = indicators[::-1].rolling(saaf_periods, min_periods=1).mean()[::-1]
            rolling_count.rename(feature, inplace=True)
            saaf_quartiles.append(rolling_count)

    saaf_quartile_df = pandas.concat(saaf_quartiles, axis=1).reindex(data.index, method="nearest")

    # convert to simple rolling mean
    saaf_data = roll(saaf_data, -saaf_periods)

    # SAAF rolling stddev, took top 2 from ElasticNet
    for num_days in [1, 8]:
        saaf_data["SAAF_stddev_{}d".format(num_days)] = saaf_data[["sx", "sy", "sz", "sa"]].rolling(num_days * 24 * saaf_periods).std().fillna(method="bfill").sum(axis=1)
    saaf_data = saaf_data.reindex(data.index, method="nearest").fillna(method="bfill")

    longterm_data = longterm_data.reindex(data.index, method="nearest")

    data = pandas.concat([data, saaf_data, longterm_data, dmop_data, event_data, event_sampled_df.reindex(data.index, method="nearest"), saaf_quartile_df], axis=1)
    assert isinstance(data, pandas.DataFrame)

    if filter_null_power:
        previous_size = data.shape[0]
        data = data[data.NPWD2532.notnull()]
        if data.shape[0] < previous_size:
            logger.info("Reduced data from {:,} rows to {:,}".format(previous_size, data.shape[0]))

    data["days_in_space"] = (data.index - pandas.datetime(year=2003, month=6, day=2)).days

    if derived_features:
        for col in [c for c in data.columns if "EVTF_IN_MRB" in c]:
            add_transformation_feature(data, col, "gradient")
        add_transformation_feature(data, "FTL_EARTH_rolling_next1h", "gradient")
        add_transformation_feature(data, "DMOP_event_counts", "log", drop=True)
        add_transformation_feature(data, "DMOP_event_counts_rolling_2h", "gradient", drop=True)
        add_transformation_feature(data, "occultationduration_min", "log", drop=True)

        add_transformation_feature(data, "sa", "log", drop=True)
        add_transformation_feature(data, "sy", "log", drop=True)

        # # various crazy rolling features
        add_lag_feature(data, "EVTF_IN_MAR_UMBRA_rolling_next1h", 50, "50")
        add_lag_feature(data, "EVTF_IN_MRB_/_RANGE_06000KM_rolling_next1h", 1600, "1600")
        add_lag_feature(data, "EVTF_event_counts_rolling_5h", 50, "50")
        add_lag_feature(data, "FTL_NADIR_rolling_next1h", 400, "400")

    logger.info("DataFrame shape %s", data.shape)
    return data

def compute_upper_bounds(dataframe):
    dataframe = dataframe[[c for c in dataframe.columns if c.startswith("NPWD")]]

    for interval in "7D 1D 12H 6H 2H 1H 30M".split():
        downsampled_data = dataframe.resample(interval).mean()
        upsampled_data = downsampled_data.reindex(dataframe.index, method="pad")

        print("RMS with {} approximation: {:.3f}".format(interval, helpers.sk._rms_error(dataframe, upsampled_data)))


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


def make_rnn(history_file=None, time_steps=4, non_negative=False):
    """Make a recurrent neural network with reasonable default args for this task"""
    model = helpers.neural.RnnRegressor(learning_rate=7e-4,
                                        num_units=50,
                                        time_steps=time_steps,
                                        batch_size=64,
                                        num_epochs=300,
                                        verbose=0,
                                        input_noise=0.05,
                                        input_dropout=0.02,
                                        early_stopping=True,
                                        recurrent_dropout=0.65,
                                        dropout=0.5,
                                        val=0.1,
                                        assert_finite=False,
                                        pretrain=True,
                                        non_negative=non_negative,
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


def load_split_data(args, data_loader=load_data, split_type="timecv"):
    """Load the data, compute cross-validation splits, scale the inputs, etc. Returns a DataSet object"""
    data = data_loader(args.training_dir, resample_interval=args.resample, filter_null_power=True)

    # cross validation by year
    if split_type == "timecv":
        splits = helpers.sk.TimeCV(data.shape[0], 10, min_training=0.7)
    elif split_type == "years":
        splits = sklearn.cross_validation.LeaveOneLabelOut(data["file_number"])
    elif split_type == "alex":
        splits = helpers.sk.WraparoundTimeCV(data.shape[0], 4, 3)
    else:
        raise ValueError("split_type={} unknown".format(split_type))

    split_map = {
        "last 30%": helpers.sk.TimeCV(data.shape[0], 10, min_training=0.7),
        "years": sklearn.cross_validation.LeaveOneLabelOut(data["file_number"]),
        "alexcv": helpers.sk.WraparoundTimeCV(data.shape[0], 4, 3)
    }
    # splits = sklearn.cross_validation.KFold(train_data.shape[0], 7, shuffle=False)
    # splits = sklearn.cross_validation.LeaveOneLabelOut(data["file_number"])

    X, Y = separate_output(data)
    if args.verify:
        verify_data(X, Y, None)
        compute_cross_validation_fairness(X.values, X.columns, Y.values, Y.columns, splits)

    if args.extra_analysis:
        X.info()
        print(X.describe())
        compute_upper_bounds(data)

    dataset = helpers.general.DataSet(X.values, Y.values, splits, X.columns, Y.columns, Y.index, split_map=split_map)
    return dataset


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
        for split_name, split in dataset.split_map:
            scores = sklearn.cross_validation.cross_val_score(model, dataset.inputs, dataset.outputs, scoring=rms_error, cv=split, n_jobs=n_jobs)
            print("[CV={}] {}: {:.4f} +/- {:.4f}".format(split_name, helpers.sk.get_model_name(model), -scores.mean(), scores.std()))
            helpers.general.get_function_logger().info("[CV={}] {}: {:.4f} +/- {:.4f}".format(split_name, helpers.sk.get_model_name(model), -scores.mean(), scores.std()))


def separate_output(df, num_outputs=None):
    logger = helpers.general.get_function_logger()
    df = df.drop("file_number", axis=1)

    Y = df[[col for col in df.columns if col.startswith("NPWD")]]
    if num_outputs:
        scores = collections.Counter({col: Y[col].mean() + Y[col].std() for col in Y.columns})
        Y = Y[[col for col, _ in scores.most_common(num_outputs)]]

    X = df[[col for col in df.columns if not col.startswith("NPWD")]]
    logger.info("X, Y shapes %s %s", X.shape, Y.shape)
    return X, Y


if __name__ == "__main__":
    sys.exit(main())