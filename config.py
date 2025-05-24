class Config:
    # Prefixes
    _ml_prefix = "./ml/"
    _ml_data_prefix = "./ml/data/"
    _ml_wrigths_prefix = "./ml/wrigths/"

    # File paths
    MODEL_PATH: str = f"{_ml_prefix}weight/random_forest.pkl"

    RAW_DATASOURCE_PATH: str = f"{_ml_prefix}data/"
    RAW_DATASOURCE_FILE: str = f"{RAW_DATASOURCE_PATH}california_housing.csv"

    Y_TRAIN_PATH: str = f"{_ml_data_prefix}y_test.csv"
    X_TRAIN_PATH: str = f"{_ml_data_prefix}X_test.csv"
    X_TEST_PATH: str = f"{_ml_data_prefix}X_test.csv"
    Y_TEST_PATH: str = f"{_ml_data_prefix}y_test.csv"

    RF_MODEL_WEIGHTS_PATH: str = _ml_wrigths_prefix
    RF_MODEL_WEIGHT_NAME: str = f"{_ml_wrigths_prefix}random_forest.pkl"
