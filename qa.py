import re

import numpy as np
import pandas as pd


def normalize_text(value):
    value = str(value).lower()
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def find_requested_column(dataframe, question):
    normalized_question = normalize_text(question)

    column_mapping = {
        normalize_text(column): column
        for column in dataframe.columns
    }

    # Check longer column names first.
    sorted_columns = sorted(
        column_mapping,
        key=len,
        reverse=True,
    )

    for normalized_column in sorted_columns:
        pattern = (
            r"(?<!\w)"
            + re.escape(normalized_column)
            + r"(?!\w)"
        )

        if re.search(
            pattern,
            normalized_question,
        ):
            return column_mapping[
                normalized_column
            ]

    return None


def answer_question(dataframe, question):
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Expected a pandas DataFrame."
        )

    if not question or not question.strip():
        return "Please enter a question."

    normalized_question = normalize_text(
        question
    )

    requested_column = find_requested_column(
        dataframe,
        normalized_question,
    )

    if any(
        phrase in normalized_question
        for phrase in [
            "how many rows",
            "number of rows",
            "row count",
            "total rows",
        ]
    ):
        return (
            f"The dataset has "
            f"{len(dataframe):,} rows."
        )

    if any(
        phrase in normalized_question
        for phrase in [
            "how many columns",
            "number of columns",
            "column count",
            "total columns",
        ]
    ):
        return (
            f"The dataset has "
            f"{dataframe.shape[1]:,} columns: "
            + ", ".join(
                map(str, dataframe.columns)
            )
            + "."
        )

    if any(
        phrase in normalized_question
        for phrase in [
            "missing",
            "null",
            "empty values",
        ]
    ):
        if requested_column is not None:
            missing_count = int(
                dataframe[
                    requested_column
                ].isna().sum()
            )

            missing_percentage = (
                dataframe[
                    requested_column
                ].isna().mean()
                * 100
            )

            return (
                f"{requested_column} has "
                f"{missing_count:,} missing values "
                f"({missing_percentage:.2f}%)."
            )

        total_missing = int(
            dataframe.isna().sum().sum()
        )

        columns_with_missing = (
            dataframe.isna()
            .sum()
            .loc[lambda values: values > 0]
            .sort_values(ascending=False)
        )

        if total_missing == 0:
            return (
                "The dataset has no missing values."
            )

        details = ", ".join(
            f"{column}={int(count)}"
            for column, count
            in columns_with_missing.items()
        )

        return (
            f"The dataset contains "
            f"{total_missing:,} missing values. "
            f"By column: {details}."
        )

    if any(
        phrase in normalized_question
        for phrase in [
            "duplicate",
            "duplicated",
        ]
    ):
        duplicate_count = int(
            dataframe.duplicated().sum()
        )

        return (
            f"The dataset contains "
            f"{duplicate_count:,} duplicate rows."
        )

    if requested_column is not None:
        series = dataframe[
            requested_column
        ]

        if pd.api.types.is_numeric_dtype(
            series
        ):
            numeric_series = (
                pd.to_numeric(
                    series,
                    errors="coerce",
                ).dropna()
            )

            if numeric_series.empty:
                return (
                    f"{requested_column} contains "
                    f"no valid numerical values."
                )

            if any(
                phrase in normalized_question
                for phrase in [
                    "average",
                    "mean",
                ]
            ):
                return (
                    f"The mean of "
                    f"{requested_column} is "
                    f"{numeric_series.mean():,.4f}."
                )

            if "median" in normalized_question:
                return (
                    f"The median of "
                    f"{requested_column} is "
                    f"{numeric_series.median():,.4f}."
                )

            if any(
                phrase in normalized_question
                for phrase in [
                    "maximum",
                    "highest",
                    "max",
                ]
            ):
                return (
                    f"The maximum value of "
                    f"{requested_column} is "
                    f"{numeric_series.max():,.4f}."
                )

            if any(
                phrase in normalized_question
                for phrase in [
                    "minimum",
                    "lowest",
                    "min",
                ]
            ):
                return (
                    f"The minimum value of "
                    f"{requested_column} is "
                    f"{numeric_series.min():,.4f}."
                )

            if any(
                phrase in normalized_question
                for phrase in [
                    "standard deviation",
                    "std",
                ]
            ):
                return (
                    f"The standard deviation of "
                    f"{requested_column} is "
                    f"{numeric_series.std():,.4f}."
                )

            if any(
                phrase in normalized_question
                for phrase in [
                    "distribution",
                    "describe",
                    "summary",
                ]
            ):
                summary = (
                    numeric_series.describe()
                    .round(4)
                    .to_dict()
                )

                return (
                    f"Summary of {requested_column}: "
                    f"{summary}"
                )

        if any(
            phrase in normalized_question
            for phrase in [
                "unique",
                "categories",
                "values",
                "most common",
            ]
        ):
            value_counts = (
                series.value_counts(
                    dropna=False
                ).head(10)
            )

            return (
                f"{requested_column} has "
                f"{series.nunique(dropna=True):,} "
                f"unique values. Most common values: "
                f"{value_counts.to_dict()}."
            )

    if "correlation" in normalized_question:
        numeric_data = dataframe.select_dtypes(
            include=np.number
        )

        if numeric_data.shape[1] < 2:
            return (
                "At least two numerical columns "
                "are required for correlation."
            )

        correlation_matrix = (
            numeric_data.corr()
        )

        upper_triangle = (
            correlation_matrix.where(
                np.triu(
                    np.ones(
                        correlation_matrix.shape
                    ),
                    k=1,
                ).astype(bool)
            )
        )

        strongest_pair = (
            upper_triangle.abs()
            .stack()
            .sort_values(
                ascending=False
            )
        )

        if strongest_pair.empty:
            return (
                "No valid correlation pair "
                "was found."
            )

        first_column, second_column = (
            strongest_pair.index[0]
        )

        correlation_value = (
            correlation_matrix.loc[
                first_column,
                second_column,
            ]
        )

        return (
            f"The strongest numerical correlation "
            f"is between {first_column} and "
            f"{second_column}: "
            f"{correlation_value:.4f}."
        )

    return (
        "I can answer questions about rows, columns, "
        "missing values, duplicates, mean, median, "
        "minimum, maximum, standard deviation, "
        "unique values, distributions and correlations. "
        "Mention an exact column name when relevant."
    )   