import pandas as pd
import numpy as np
from data_quality_checks.data_quality_checks import DataQualityChecks


class ConsistencyChecks(DataQualityChecks):
    """
    Implements consistency checks as a child of DataQualityChecks.
    """

    def metrics(self, check_type, *args, **kwargs):
        """
        Dynamically call the metrics method for the given check type.
        """
        metrics_methods = {
            "unique_identifiers": self.metrics_unique_identifiers,
            "schema_consistency": self.metrics_schema_consistency,
            "non_null": self.metrics_non_null,
            "threshold": self.metrics_threshold,
            "dynamic_threshold": self.metrics_dynamic_threshold,
            "variance": self.metrics_variance,
            "record_anomalies": self.metrics_record_anomalies,
            "non_zero_records": self.metrics_non_zero_records,
            "column_name_consistency": self.metrics_column_name_consistency,
        }

        if check_type in metrics_methods:
            return metrics_methods[check_type](*args, **kwargs)
        else:
            raise ValueError(f"Unknown check type for metrics: {check_type}")

    def rules(self, check_type, *args, **kwargs):
        """
        Dynamically call the rules method for the given check type.
        """
        rules_methods = {
            "unique_identifiers": self.rules_unique_identifiers,
            "schema_consistency": self.rules_schema_consistency,
            "non_null": self.rules_non_null,
            "threshold": self.rules_threshold,
            "dynamic_threshold": self.rules_dynamic_threshold,
            "variance": self.rules_variance,
            "record_anomalies": self.rules_record_anomalies,
            "non_zero_records": self.rules_non_zero_records,
            "column_name_consistency": self.rules_column_name_consistency,
        }

        if check_type in rules_methods:
            return rules_methods[check_type](*args, **kwargs)
        else:
            raise ValueError(f"Unknown check type for rules: {check_type}")

    # Unique Identifier Consistency
    def metrics_unique_identifiers(self, df, id_col):
        return df[id_col].duplicated().sum()

    def rules_unique_identifiers(self, df, id_col):
        duplicates = self.metrics_unique_identifiers(df, id_col)
        return duplicates == 0, f"{duplicates} duplicate identifiers found."

    # Schema Consistency
    def metrics_schema_consistency(self, df, expected_columns):
        return set(expected_columns) - set(df.columns)

    def rules_schema_consistency(self, df, expected_columns):
        missing_cols = self.metrics_schema_consistency(df, expected_columns)
        return len(missing_cols) == 0, f"Missing columns: {missing_cols}"

    # Non-Null Checks
    def metrics_non_null(self, df, columns):
        return {col: df[col].isnull().sum() for col in columns}

    def rules_non_null(self, df, columns):
        null_counts = self.metrics_non_null(df, columns)
        invalid_cols = {col: count for col, count in null_counts.items() if count > 0}
        return len(invalid_cols) == 0, f"Null values found in columns: {invalid_cols}"

    # Threshold Limits
    def metrics_threshold(self, df, column, min_val, max_val):
        return ((df[column] < min_val) | (df[column] > max_val)).sum()

    def rules_threshold(self, df, column, min_val, max_val):
        outliers = self.metrics_threshold(df, column, min_val, max_val)
        return outliers == 0, f"{outliers} values are outside the range [{min_val}, {max_val}]."

    # Dynamic Thresholds
    def metrics_dynamic_threshold(self, df, column, reference_value, tolerance):
        return (
            (df[column] < reference_value * (1 - tolerance)) |
            (df[column] > reference_value * (1 + tolerance))
        ).sum()

    def rules_dynamic_threshold(self, df, column, reference_value, tolerance):
        outliers = self.metrics_dynamic_threshold(df, column, reference_value, tolerance)
        return outliers == 0, f"{outliers} values exceed dynamic thresholds."

    # Variance Checks
    def metrics_variance(self, df, column):
        return df[column].var()

    def rules_variance(self, df, column, max_variance):
        variance = self.metrics_variance(df, column)
        return variance <= max_variance, f"Variance ({variance}) exceeds the maximum allowed ({max_variance})."

    # Anomaly Detection on Number of Records
    def metrics_record_anomalies(self, record_counts, lookback_period=30):
        mean = np.mean(record_counts[-lookback_period:])
        std_dev = np.std(record_counts[-lookback_period:])
        return mean, std_dev

    def rules_record_anomalies(self, record_counts, current_count, lookback_period=30):
        mean, std_dev = self.metrics_record_anomalies(record_counts, lookback_period)
        lower_limit = mean - 3 * std_dev
        upper_limit = mean + 3 * std_dev
        return lower_limit <= current_count <= upper_limit, (
            f"Record count {current_count} is outside 3 standard deviations "
            f"[{lower_limit}, {upper_limit}] of the mean."
        )

    # Check Number of Records Greater Than Zero
    def metrics_non_zero_records(self, df):
        return len(df)

    def rules_non_zero_records(self, df):
        record_count = self.metrics_non_zero_records(df)
        return record_count > 0, "No records found in the dataset."

    # Check Column Names Consistency
    def metrics_column_name_consistency(self, df, historical_columns):
        return set(historical_columns) - set(df.columns)

    def rules_column_name_consistency(self, df, historical_columns):
        inconsistent_cols = self.metrics_column_name_consistency(df, historical_columns)
        return len(inconsistent_cols) == 0, f"Inconsistent column names: {inconsistent_cols}"
