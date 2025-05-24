# tests/test_data_parser.py
import pandas as pd
import pytest

from data_parser import DataParser
from ml.data_parser import data_parser
from ml.common.schemas.california_housing_model import CaliforniaHousingModel

def test_download_and_cache(tmp_path, monkeypatch):
    # ensure no cache exists
    cache_file = tmp_path / "california_housing.csv"
    assert not cache_file.exists()

    # monkeypatch fetch to record that it's called
    called = {"flag": False}
    import sklearn.datasets as ds
    def fake_fetch(*args, **kwargs):
        called["flag"] = True
        return ds.fetch_california_housing(as_frame=True)
    monkeypatch.setattr(ds, "fetch_california_housing", fake_fetch)

    # first initialization → should download & write cache
    parser = DataParser(cache_path=str(cache_file))
    assert called["flag"], "Dataset should be fetched on first run"
    assert cache_file.exists(), "Cache file must be created"
    assert not parser.df.empty

    # parsing produces a valid Pydantic model
    inst = parser.parse_sample()
    assert isinstance(inst, CaliforniaHousingModel)
    # check field values match the first row
    first = parser.df.iloc[0].to_dict()
    for k, v in first.items():
        assert getattr(inst, k) == pytest.approx(v)


def test_load_from_cache_only(tmp_path, monkeypatch):
    # create a minimal dummy CSV cache
    cache_file = tmp_path / "california_housing.csv"
    cols = ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]
    dummy = pd.DataFrame([{c: i + 0.1 for i, c in enumerate(cols)}])
    dummy.to_csv(cache_file, index=False)

    # monkeypatch fetch to fail if called
    def fail_fetch(*args, **kwargs):
        pytest.fail("fetch_california_housing should NOT be called when cache exists")
    monkeypatch.setattr(data_parser, "fetch_california_housing", fail_fetch)

    parser = DataParser(cache_path=str(cache_file))
    # it should load our dummy cache
    pd.testing.assert_frame_equal(parser.df, dummy)

    # parse_sample still works
    inst = parser.parse_sample()
    assert isinstance(inst, CaliforniaHousingModel)
    for c in cols:
        assert getattr(inst, c) == pytest.approx(dummy.iloc[0][c])
