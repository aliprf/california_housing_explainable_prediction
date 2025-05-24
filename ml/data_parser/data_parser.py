import os
import pandas as pd
from sklearn.datasets import fetch_california_housing

from ml.common.schemas.california_housing_model import CaliforniaHousingModel
from config import Config

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataParser:
    def __init__(self, cache_path: str = Config.RAW_DATASOURCE_FILE):
        self.cache_path = cache_path
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        if os.path.exists(self.cache_path):
            logger.info(f"cache df found: {self.cache_path}")
            self.df = pd.read_csv(self.cache_path)
        else:
            logger.info(f"no cache in: {self.cache_path} || Downloading")
            self._download_and_cache()

    def _download_and_cache(self):
        data_bunch = fetch_california_housing(as_frame=True)
        self.df = data_bunch.frame
        self.df.to_csv(self.cache_path, index=False)

    def get_sample_instance(self) -> dict:
        return self.df.iloc[0].to_dict()

    def parse_sample(self) -> CaliforniaHousingModel:
        sample = self.get_sample_instance()
        return CaliforniaHousingModel(**sample)


if __name__ == "__main__":
    parser = DataParser()
    print(parser.df.head())
    sample = parser.parse_sample()
    print("Parsed sample model:", sample)
