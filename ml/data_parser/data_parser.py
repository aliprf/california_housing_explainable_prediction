import logging
import os

import pandas as pd
from sklearn.datasets import fetch_california_housing

from config import Config
from ml.common.schemas.california_housing_model import CaliforniaHousingModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataParser:
    def __init__(self, cache_path: str = Config.RAW_DATASOURCE_FILE) -> None:
        """Initialize the DataParser.

        Args:
            cache_path (str, optional): Path to the cache file. Defaults to Config.
            RAW_DATASOURCE_FILE.

        """
        self.cache_path = cache_path
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        if os.path.exists(self.cache_path):
            msg = f"cache found: {self.cache_path}"
            logger.info(msg)
            self.df = pd.read_csv(self.cache_path)
        else:
            msg = f"cache not found: {self.cache_path}"
            logger.info(msg)
            self._download_and_cache()

    def _download_and_cache(self) -> None:
        data_bunch = fetch_california_housing(as_frame=True)
        self.df = data_bunch.frame
        self.df.to_csv(self.cache_path, index=False)

    def get_sample_instance(self) -> dict:
        """Get a sample instance from the DataFrame.

        Returns:
            dict: A dictionary representation of the sample instance.

        """
        return self.df.iloc[0].to_dict()

    def parse_sample(self) -> CaliforniaHousingModel:
        """Parse a sample instance into a Pydantic model.

        Returns:
            CaliforniaHousingModel: A Pydantic model representation of
                the sample instance.

        """
        sample = self.get_sample_instance()
        return CaliforniaHousingModel(**sample)


if __name__ == "__main__":
    parser = DataParser()
    logger.info(parser.df.head())
    sample = parser.parse_sample()
    msg = f"Sample instance: {sample}"
    logger.info(msg)
