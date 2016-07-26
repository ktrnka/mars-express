import unittest
from operator import itemgetter

import fixed_features
from loaders import parse_cut_feature, select_features
from test_feature_selection import split_feature_name, diversify


class FeatureNameTests(unittest.TestCase):
    def test_names(self):
        self.assertSequenceEqual(("DMOP_event_counts", None), split_feature_name("DMOP_event_counts"))
        self.assertSequenceEqual(("DMOP_event_counts", "5h"), split_feature_name("DMOP_event_counts_5h"))
        self.assertSequenceEqual(("DMOP_event_counts", "next5h"), split_feature_name("DMOP_event_counts_next5h"))
        self.assertSequenceEqual(("DMOP_event_counts", None), split_feature_name("DMOP_event_counts_log"))
        self.assertSequenceEqual(("DMOP_event_counts", "next5h"), split_feature_name("DMOP_event_counts_next5h"))
        self.assertSequenceEqual(("DMOP_event_counts", "next5h"), split_feature_name("DMOP_event_counts_next5h_log"))
        self.assertSequenceEqual(("DMOP_event_counts", "next5h"), split_feature_name("DMOP_event_counts_log_next5h"))
        self.assertSequenceEqual(("EVTF_event_counts", "16h"), split_feature_name("EVTF_event_counts_rolling_16h"))
        self.assertEqual("EVTF_event_counts", split_feature_name("EVTF_event_counts_rolling_16h_rolling_20")[0])
        self.assertEqual(split_feature_name("LT_eclipseduration_min")[0], split_feature_name("LT_eclipseduration_min_rolling_1d")[0])
        self.assertEqual(("EVTF_IN_MRB_/_RANGE_06000KM", "1h"), split_feature_name("EVTF_IN_MRB_/_RANGE_06000KM_rolling_1h_rolling_1600"))

    def test_diversity(self):
        feature_names = "DMOP_event_counts DMOP_event_counts_5h EVTF_event_counts_rolling_16h EVTF_event_counts_rolling_2h".split()
        feature_scores = [1.0, 1.1, 0.3, 0.81]

        # normal pick 2
        selected_features = sorted(zip(feature_names, feature_scores), key=itemgetter(1), reverse=True)[:2]
        self.assertSequenceEqual(["DMOP_event_counts_5h", "DMOP_event_counts"], map(itemgetter(0), selected_features))

        # diversity pick 2
        diversified_scores = diversify(feature_names, feature_scores)
        selected_features = sorted(zip(feature_names, diversified_scores), key=itemgetter(1), reverse=True)[:2]
        self.assertSequenceEqual(["DMOP_event_counts_5h", "EVTF_event_counts_rolling_2h"], map(itemgetter(0), selected_features))

        # diversity only matters when they're overlapping
        feature_names = "DMOP_event_counts_next5h DMOP_event_counts_5h EVTF_event_counts_rolling_16h EVTF_event_counts_rolling_2h".split()

        diversified_scores = diversify(feature_names, feature_scores)
        selected_features = sorted(zip(feature_names, diversified_scores), key=itemgetter(1), reverse=True)[:2]
        self.assertSequenceEqual(["DMOP_event_counts_5h", "DMOP_event_counts_next5h"], map(itemgetter(0), selected_features))

class TestSaafFeatureNameParser(unittest.TestCase):
    def test_name(self):
        examples = ["sz__(124.7, 179.735]_rolling_next48h", "sz__(124.7, 179.735]_rolling_48h", "sz__(124.7, 179.735]"]

        base, lower, upper = parse_cut_feature(examples[0])

        self.assertEqual("sz", base)
        self.assertEqual(124.7, lower)
        self.assertEqual(179.735, upper)

        base, lower, upper = parse_cut_feature(examples[1])

        self.assertEqual("sz", base)
        self.assertEqual(124.7, lower)
        self.assertEqual(179.735, upper)

    def test_feature_sel(self):
        selected_features = select_features(fixed_features.feature_weights, 10)
        self.assertEqual(selected_features[0], fixed_features.feature_weights[0][0])
        self.assertEqual(selected_features[1], fixed_features.feature_weights[1][0])
        self.assertEqual(10, len(selected_features))