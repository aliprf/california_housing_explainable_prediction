import os
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.datasets import fetch_california_housing

from ml.common.schemas.california_housing_model import CaliforniaHousingModel
from ml.data_parser.data_parser import DataParser


@pytest.fixture
def tmp_cache_path(tmp_path):
    return tmp_path / "housing.csv"


def test_download_and_cache_creates_file(tmp_cache_path):
    # File shouldn't exist before
    assert not tmp_cache_path.exists()

    parser = DataParser(cache_path=str(tmp_cache_path))

    # ✅ Path exists
    assert os.path.exists(tmp_cache_path)

    # ✅ DataFrame loaded
    assert isinstance(parser.df, pd.DataFrame)
    assert not parser.df.empty

def test_get_sample_instance(tmp_cache_path):
    parser = DataParser(cache_path=str(tmp_cache_path))
    sample = parser.get_sample_instance()

    assert isinstance(sample, dict)
    assert "MedInc" in sample  # Example column from the dataset


def test_parse_sample_returns_basemodel(tmp_cache_path):
    parser = DataParser(cache_path=str(tmp_cache_path))
    model = parser.parse_sample()

    assert isinstance(model, CaliforniaHousingModel)


def test_all_rows_valid_pydantic(tmp_cache_path):
    parser = DataParser(cache_path=str(tmp_cache_path))

    for _, row in parser.df.iterrows():
        instance = CaliforniaHousingModel(**row.to_dict())
        assert isinstance(instance, CaliforniaHousingModel)
