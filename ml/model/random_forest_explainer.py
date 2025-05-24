import os
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.ensemble import RandomForestRegressor

class RandomForestExplainer:
    def __init__(
            self,
            model_path: str = "./ml/weights/random_forest.pkl", 
            x_test_path: str = "./ml/weights/X_test.csv",
            y_test_path: str = "./ml/weights/y_test.csv"):
        self.model_path = model_path
        self.x_test_path = x_test_path
        self.y_test_path = y_test_path

        self.model: RandomForestRegressor | None = None
        self.X_test: pd.DataFrame | None= None
        self.y_test: pd.Series | None= None
        self.explainer = None
        self.shap_values = None

        self._load_model_and_data()

    def _load_model_and_data(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = joblib.load(self.model_path)
        self.X_test = pd.read_csv(self.x_test_path)
        self.y_test = pd.read_csv(self.y_test_path).squeeze()

        self.explainer = shap.Explainer(self.model, self.X_test)
        self.shap_values = self.explainer(self.X_test)

    def get_feature_importance(self):
        importances = self.model.feature_importances_
        feature_names = self.X_test.columns.tolist()
        return {
            "feature_names": feature_names,
            "importances": importances.tolist()
        }

    def get_shap_explanation(self, sample_indices: list = [0]):
        if not sample_indices:
            raise ValueError("No sample indices provided")

        shap_output = {
            "feature_names": self.X_test.columns.tolist(),
            "expected_value": float(self.shap_values.base_values[0]),
            "shap_values": []
        }

        for idx in sample_indices:
            if idx >= len(self.X_test):
                raise IndexError(f"Sample index {idx} out of bounds")
            shap_row = self.shap_values[idx].values
            shap_output["shap_values"].append(shap_row.tolist())

        return shap_output

    def save_feature_importance_plot(self, out_path="./ml/weights/feature_importance.png"):
        data = self.get_feature_importance()
        indices = np.argsort(data['importances'])[::-1]
        sorted_names = [data['feature_names'][i] for i in indices]
        sorted_values = np.array(data['importances'])[indices]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(len(sorted_values)), sorted_values, align='center')
        plt.xticks(range(len(sorted_values)), sorted_names, rotation=45, ha="right")
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        plt.close()

    def save_shap_plot(self, sample_index: int = 0, out_path="./ml/weights/shap_plot.png"):
        if sample_index >= len(self.X_test):
            raise IndexError(f"Sample index {sample_index} out of bounds")
        shap.plots.waterfall(self.shap_values[sample_index], show=False)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

if __name__ == "__main__":
    explainer = RandomForestExplainer()
    fi = explainer.get_feature_importance()
    print("Feature Importance:")
    print(json.dumps(fi, indent=2))

    shap_exp = explainer.get_shap_explanation([0])
    print("\nSHAP Explanation:")
    print(json.dumps(shap_exp, indent=2))

    explainer.save_feature_importance_plot()
    explainer.save_shap_plot(0)
