from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )
def _df_for_new_functions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'name': ['Alex', 'Bob', 'Nancy', 'Annu'],
            'salary':[0, 0, 0, 100000],
            'age': [22, 22, 22, 22]
        }
    )

def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2

def test_has_constant_columns():
    df = _df_for_new_functions()
    summary = summarize_dataset(df)
    col_by_name = {col.name: col for col in summary.columns}
    assert col_by_name["age"].has_constant_values is True
    assert col_by_name["name"].has_constant_values is False
    assert col_by_name["salary"].has_constant_values is False

def test_too_many_zeroes():
    df = _df_for_new_functions()
    summary = summarize_dataset(df)
    col_by_name = {col.name: col for col in summary.columns}
    assert col_by_name['salary'].big_zero_share is True
    assert col_by_name['age'].big_zero_share is False
