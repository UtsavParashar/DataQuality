import pytest
import pandas as pd
from data_quality_checks.consistency_checks import ConsistencyChecks


@pytest.fixture
def sample_data():
    """
    Sample DataFrame for testing.
    """
    data = {
        "id": [1, 2, 3, 4, 4],
        "name": ["Alice", "Bob", "Charlie", "David", None],
        "value": [10, 20, 30, 40, 50],
        "timestamp": pd.date_range("2023-01-01", periods=5),
    }
    return pd.DataFrame(data)


@pytest.fixture
def historical_columns():
    """
    Provides historical column names for schema consistency testing.
    """
    return ["id", "name", "value", "timestamp"]


@pytest.fixture
def record_counts():
    """
    Sample record counts for anomaly detection.
    """
    return pd.Series([100, 110, 105, 95, 90, 115, 120, 100, 105, 110, 95, 105, 100])


# Test 1: Unique Identifier Consistency
def test_unique_identifier_consistency(sample_data):
    cc = ConsistencyChecks()
    metric = cc.metrics("unique_identifiers", sample_data, "id")
    rule, message = cc.rules("unique_identifiers", sample_data, "id")
    assert metric == 1, "Expected 1 duplicate identifier."
    assert not rule, message


# Test 2: Schema Consistency
def test_schema_consistency(sample_data, historical_columns):
    cc = ConsistencyChecks()
    metric = cc.metrics("schema_consistency", sample_data, historical_columns)
    rule, message = cc.rules("schema_consistency", sample_data, historical_columns)
    assert len(metric) == 0, f"Unexpected missing columns: {metric}"
    assert rule, message


# Test 3: Non-Null Checks
def test_non_null_checks(sample_data):
    cc = ConsistencyChecks()
    columns = ["name"]
    metric = cc.metrics("non_null", sample_data, columns)
    rule, message = cc.rules("non_null", sample_data, columns)
    assert metric["name"] == 1, "Expected 1 null value in 'name'."
    assert not rule, message


# Test 4: Threshold Limits
def test_threshold_limits(sample_data):
    cc = ConsistencyChecks()
    metric = cc.metrics("threshold", sample_data, "value", 10, 50)
    rule, message = cc.rules("threshold", sample_data, "value", 10, 50)
    assert metric == 0, "Expected no values outside threshold limits."
    assert rule, message


# Test 5: Dynamic Thresholds
def test_dynamic_thresholds(sample_data):
    cc = ConsistencyChecks()
    reference_value = 30
    tolerance = 0.5  # 50% tolerance
    metric = cc.metrics("dynamic_threshold", sample_data, "value", reference_value, tolerance)
    rule, message = cc.rules("dynamic_threshold", sample_data, "value", reference_value, tolerance)
    assert metric == 2, f"Expected 2 values outside dynamic thresholds; found {metric}."
    assert not rule, message


# Test 6: Variance Checks
def test_variance_checks(sample_data):
    cc = ConsistencyChecks()
    metric = cc.metrics("variance", sample_data, "value")
    rule, message = cc.rules("variance", sample_data, "value", threshold=50)
    assert metric <= 50, "Variance should not exceed the threshold."
    assert rule, message


# Test 7: Anomaly Detection on Number of Records
def test_anomaly_detection_on_records(record_counts):
    cc = ConsistencyChecks()
    metric = cc.metrics("record_anomalies", record_counts)
    rule, message = cc.rules("record_anomalies", record_counts)
    assert metric == 0, "Expected no anomalies in record counts."
    assert rule, message


# Test 8: Record Count Greater Than Zero
def test_record_count_greater_than_zero(sample_data):
    cc = ConsistencyChecks()
    metric = cc.metrics("record_count_greater_than_zero", sample_data)
    rule, message = cc.rules("record_count_greater_than_zero", sample_data)
    assert metric > 0, "Record count should be greater than zero."
    assert rule, message


# Test 9: Column Names Consistency
def test_column_names_consistency(sample_data, historical_columns):
    cc = ConsistencyChecks()
    metric = cc.metrics("column_names_consistency", sample_data, historical_columns)
    rule, message = cc.rules("column_names_consistency", sample_data, historical_columns)
    assert len(metric) == 0, "Column names should match the historical schema."
    assert rule, message
