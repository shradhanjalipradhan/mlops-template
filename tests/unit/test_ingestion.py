"""Unit tests for data ingestion and cleaning."""
import pytest
import pandas as pd
import numpy as np
from src.data.ingestion import clean, split, load_raw
from src.data.validation import validate


class TestClean:
    def test_removes_duplicates(self, raw_df):
        duped = pd.concat([raw_df, raw_df.iloc[:10]], ignore_index=True)
        result = clean(duped)
        assert len(result) == len(raw_df)

    def test_drops_out_of_range_age(self, raw_df):
        dirty = raw_df.copy()
        dirty.loc[0, "age"] = 10.0   # below 18
        dirty.loc[1, "age"] = 200.0  # above 100
        result = clean(dirty)
        assert result["age"].between(18, 100).all()

    def test_drops_null_rows_in_required_columns(self, raw_df):
        dirty = raw_df.copy()
        dirty.loc[0, "monthly_charge"] = np.nan
        result = clean(dirty)
        assert result["monthly_charge"].notna().all()

    def test_raises_on_missing_column(self, raw_df):
        broken = raw_df.drop(columns=["churned"])
        with pytest.raises(ValueError, match="Missing required columns"):
            clean(broken)


class TestValidate:
    def test_passes_on_valid_df(self, raw_df):
        cleaned = clean(raw_df)
        assert validate(cleaned) is True

    def test_fails_on_too_few_rows(self, raw_df):
        tiny = raw_df.iloc[:5].copy()
        with pytest.raises(ValueError, match="Too few rows"):
            validate(tiny)

    def test_fails_on_non_binary_target(self, raw_df):
        dirty = raw_df.copy()
        dirty.loc[0, "churned"] = 99
        with pytest.raises(ValueError, match="binary"):
            validate(dirty)


class TestSplit:
    def test_split_sizes_sum_to_total(self, raw_df):
        cleaned = clean(raw_df)
        train, val, test = split(cleaned, test_size=0.2, val_size=0.1)
        assert len(train) + len(val) + len(test) == len(cleaned)

    def test_no_overlap_between_splits(self, raw_df):
        cleaned = clean(raw_df).reset_index(drop=True)
        train, val, test = split(cleaned)
        idx_sets = [set(train.index), set(val.index), set(test.index)]
        assert idx_sets[0].isdisjoint(idx_sets[1])
        assert idx_sets[0].isdisjoint(idx_sets[2])
        assert idx_sets[1].isdisjoint(idx_sets[2])

    def test_stratification_preserves_class_ratio(self, raw_df):
        cleaned = clean(raw_df)
        original_rate = cleaned["churned"].mean()
        train, val, test = split(cleaned)
        for split_df in [train, val, test]:
            ratio = split_df["churned"].mean()
            assert abs(ratio - original_rate) < 0.1
