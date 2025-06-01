import logging
import os
from random import randint
from typing import Union

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import train_test_split

from config import Config
from ml.common.schemas.california_housing_model import CaliforniaHousingModel
from ml.data_parser.data_parser import DataParser

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(handler)

class RandomForestPrediction:
    def __init__(self, model_path: str = Config.MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.data_parser = DataParser()
        self.df = self.data_parser.df

        # Store train/test split as attributes to reuse later
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(
        self, test_size: float = 0.25, random_state: int = randint(1, 100), **rf_kwargs
    ):
        logger.info("Training Random Forest model.")
        X = self.df.drop("MedHouseVal", axis=1)
        y = self.df["MedHouseVal"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )
        logger.info(
            f"Training samples: {len(self.X_train)}, Test samples: {len(self.X_test)}",
        )

        self.model = RandomForestRegressor(**rf_kwargs)
        self.model.fit(self.X_train, self.y_train)

        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        logger.info(f"Random Forest trained. MSE on test set: {mse:.4f}")
        return self.model

    def save_model(self):
        if self.model is not None:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            # Save test set as well for later evaluation

            if self.X_test is None or self.y_test is None:
                raise ValueError("test ds was not found")

            self.X_test.to_csv(Config.X_TEST_PATH, index=False)
            self.y_test.to_csv(Config.Y_TEST_PATH, index=False)
            logger.info("Test set saved for later evaluation.")
        else:
            logger.error("No model to save. Train a model first.")
            raise ValueError("No model to save. Train a model first.")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")

            # Load the saved test set
            self.X_test = pd.read_csv(Config.X_TEST_PATH)
            self.y_test = pd.read_csv(
                Config.Y_TEST_PATH,
            ).squeeze()
            logger.info("Test set loaded for evaluation.")
        else:
            logger.error(f"No model file found at {self.model_path}")
            raise FileNotFoundError(f"No model file found at {self.model_path}")

    def predict_df(self, X: pd.DataFrame):
        if self.model is None:
            logger.error("Model is not loaded or trained.")
            raise ValueError("Model is not loaded or trained.")
        logger.info(f"Making predictions for {len(X)} samples.")
        return self.model.predict(X)

    def predict(
        self,
        input_data: Union[CaliforniaHousingModel, list[CaliforniaHousingModel]],
    ):
        if self.model is None:
            logger.error("Model is not loaded or trained.")
            raise ValueError("Model is not loaded or trained.")

        # Normalize input to list
        if isinstance(input_data, CaliforniaHousingModel):
            input_data = [input_data]

        # Convert list of BaseModel instances to DataFrame
        data_dicts = [item.model_dump() for item in input_data]
        X = pd.DataFrame(data_dicts)

        logger.info(f"Making predictions for {len(X)} samples.")
        preds = self.model.predict(X)
        return preds

    def evaluate(
        self,
        y_true,
        y_pred,
        threshold: float | None = None,
        save_path: str = f"{Config.RF_MODEL_WEIGHTS_PATH}evaluation_metrics.txt",
    ):
        if threshold is None:
            threshold = pd.Series(y_true).median()
        y_true_bin = (y_true > threshold).astype(int)
        y_pred_bin = (y_pred > threshold).astype(int)

        f1 = f1_score(y_true_bin, y_pred_bin)
        precision = precision_score(y_true_bin, y_pred_bin)
        recall = recall_score(y_true_bin, y_pred_bin)

        logger.info(
            f"Evaluation metrics (threshold={threshold:.2f}): F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}",
        )

        # Prepare metrics text
        metrics_text = (
            f"Evaluation metrics (threshold={threshold:.2f}):\n"
            f"F1 Score: {f1:.4f}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall: {recall:.4f}\n"
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save to text file
        with open(save_path, "w") as f:
            f.write(metrics_text)

        logger.info(f"Saved evaluation metrics to {save_path}")


if __name__ == "__main__":
    trainer = RandomForestPrediction()

    # Step 1: Train and save model
    trainer.train()
    trainer.save_model()

    if trainer.X_test is None:
        raise ValueError("trainer.X_test is None!!")
    y_pred = trainer.predict_df(trainer.X_test)
    trainer.evaluate(trainer.y_test, y_pred)

    # Step 2: Load weights and predict on a sample input
    trainer.load_model()
    sample_dict = trainer.data_parser.get_sample_instance()
    sample_input: CaliforniaHousingModel = CaliforniaHousingModel(**sample_dict)
    prediction = trainer.predict(sample_input)

    logger.info("Predicted Median House Values:")
    for p in prediction:
        logger.info(f" prediction: {p:.4f}")
