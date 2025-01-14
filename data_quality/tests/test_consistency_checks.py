import pytest
import pandas as pd
import sys

from data_quality.checks.consistency import ConsistencyChecks

class TestConsistencyChecks:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.cc = ConsistencyChecks()

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 4],
                "name": ["Alice", "Bob", "Charlie", "David", None],
                "value": [10, 20, 30, 40, 50],
                "timestamp": pd.date_range("2023-01-01", periods=5),
            }
        )

    @pytest.fixture
    def historical_columns(self):
        return ["id", "name", "value", "timestamp"]

    @pytest.fixture
    def record_counts(self):
        return pd.Series([100, 110, 105, 95, 90, 115, 120, 100, 105, 110, 95, 105, 100])

    def test_unique_identifier_consistency(self, sample_data):
        metric = self.cc.metrics("unique_identifiers", sample_data, "id")
        rule, message = self.cc.rules("unique_identifiers", sample_data, "id")
        assert metric == 1, "Expected 1 duplicate identifier."
        assert not rule, message

    def test_schema_consistency(self, sample_data, historical_columns):
        metric = self.cc.metrics("schema_consistency", sample_data, historical_columns)
        rule, message = self.cc.rules("schema_consistency", sample_data, historical_columns)
        assert len(metric) == 0, f"Unexpected missing columns: {metric}"
        assert rule, message

    def test_non_null_checks(self, sample_data):
        metric = self.cc.metrics("non_null", sample_data, ["name"])
        rule, message = self.cc.rules("non_null", sample_data, ["name"])
        assert metric["name"] == 1, "Expected 1 null value in 'name'."
        assert not rule, message

    def test_threshold_limits(self, sample_data):
        metric = self.cc.metrics("threshold", sample_data, "value", 10, 50)
        rule, message = self.cc.rules("threshold", sample_data, "value", 10, 50)
        assert metric == 0, "Expected no values outside threshold limits."
        assert rule, message

    def test_dynamic_thresholds(self, sample_data):
        metric = self.cc.metrics("dynamic_threshold", sample_data, "value", 30, 0.5)
        rule, message = self.cc.rules("dynamic_threshold", sample_data, "value", 30, 0.5)
        assert metric == 2, f"Expected 2 values outside dynamic thresholds; found {metric}."
        assert not rule, message

    def test_variance_checks(self, sample_data):
        metric = self.cc.metrics("variance", sample_data, "value")
        rule, message = self.cc.rules("variance", sample_data, "value", threshold=50)
        assert metric <= 50, "Variance should not exceed the threshold."
        assert rule, message

    def test_anomaly_detection_on_records(self, record_counts):
        metric = self.cc.metrics("record_anomalies", record_counts)
        rule, message = self.cc.rules("record_anomalies", record_counts)
        assert metric == 0, "Expected no anomalies in record counts."
        assert rule, message

    def test_record_count_greater_than_zero(self, sample_data):
        metric = self.cc.metrics("record_count_greater_than_zero", sample_data)
        rule, message = self.cc.rules("record_count_greater_than_zero", sample_data)
        assert metric > 0, "Record count should be greater than zero."
        assert rule, message

    def test_column_names_consistency(self, sample_data, historical_columns):
        metric = self.cc.metrics("column_names_consistency", sample_data, historical_columns)
        rule, message = self.cc.rules("column_names_consistency", sample_data, historical_columns)
        assert len(metric) == 0, "Column names should match the historical schema."
        assert rule, message
