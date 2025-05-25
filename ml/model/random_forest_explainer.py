from html import escape
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import PartialDependenceDisplay
from sklearn.tree import DecisionTreeRegressor

from config import Config

matplotlib.use("Agg")  # For headless environments

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(handler)


class RandomForestExplainer:
    def __init__(
        self,
        model_path: str = Config.MODEL_PATH,
        shap_cache_path: str = Config.SHAP_CACHE_PATH,
        x_test_path: str = Config.X_TEST_PATH,
        y_test_path: str = Config.Y_TEST_PATH,
    ) -> None:
        self.model_path = model_path
        self.shap_cache_path = shap_cache_path
        self.x_test_path = x_test_path
        self.y_test_path = y_test_path

        try:
            if not Path(self.model_path).exists():
                msg = f"Model file not found: {self.model_path}"
                raise FileNotFoundError(msg)  # noqa: TRY301

            self.model = joblib.load(self.model_path)
            self.X_test = pd.read_csv(self.x_test_path)
            self.y_test = pd.read_csv(self.y_test_path).squeeze()

            self.explainer = shap.Explainer(self.model, self.X_test)

            if Path(self.shap_cache_path).exists():
                msg = f"Loading cached SHAP values from {self.shap_cache_path}"
                logger.info(msg)
                self.shap_values = joblib.load(self.shap_cache_path)
            else:
                # calc
                logger.info("Calculating SHAP values...")
                self.shap_values = self.explainer(self.X_test, check_additivity=False)
                # save
                Path(Config.RF_MODEL_WEIGHTS_PATH).parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                joblib.dump(self.shap_values, self.shap_cache_path)
        except Exception as e:
            msg = f"Error in _load_model_and_data: {e!s}"
            logger.exception(msg)
            raise

    def get_feature_importance(
        self,
        save_name: str = f"{Config.RF_MODEL_WEIGHTS_PATH}/feature_importance.json",
    ):
        if self.X_test is None or self.model is None:
            msg = "Model or test data not loaded."
            raise ValueError(msg)

        importances = self.model.feature_importances_
        feature_names = self.X_test.columns.tolist()
        result = {"feature_names": feature_names, "importances": importances.tolist()}

        path = Path(save_name)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

        return result

    def get_shap_explanation(
        self,
        sample_indices: list[int],
        save_name: str = f"{Config.RF_MODEL_WEIGHTS_PATH}/shap_explanation.json",
    ) -> dict[str, Any]:
        if not sample_indices:
            msg = "No sample indices provided"
            raise ValueError(msg)
        if self.X_test is None or self.shap_values is None:
            msg = "Test data or SHAP values not available"
            raise ValueError(msg)

        shap_output = {
            "feature_names": self.X_test.columns.tolist(),
            "expected_value": float(self.shap_values.base_values[0]),
            "shap_values": [
                self.shap_values[i].values.tolist() for i in sample_indices
            ],
        }

        path = Path(save_name)
        with path.open("w", encoding="utf-8") as f:
            json.dump(shap_output, f, indent=4)

        return shap_output

    def save_feature_importance_plot(
        self,
        out_path: str = f"{Config.RF_MODEL_WEIGHTS_PATH}/feature_importance.png",
    ) -> None:
        data = self.get_feature_importance()
        indices = np.argsort(data["importances"])[::-1]
        sorted_names = [data["feature_names"][i] for i in indices]
        sorted_values = np.array(data["importances"])[indices]

        plt.figure(figsize=(20, 12))
        plt.title("Feature Importance")
        plt.bar(range(len(sorted_values)), sorted_values, align="center")
        plt.xticks(range(len(sorted_values)), sorted_names, rotation=45, ha="right")
        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()

    def save_shap_plot(
        self,
        sample_index: int = 0,
        out_path: str = f"{Config.RF_MODEL_WEIGHTS_PATH}/shap_plot.png",
    ) -> None:
        if self.X_test is None or self.shap_values is None:
            msg = "Test data or SHAP values not available"
            raise ValueError(msg)

        shap.plots.waterfall(self.shap_values[sample_index], show=False)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    def save_partial_dependence_plot(
        self,
        features: list,
        out_path: str = f"{Config.RF_MODEL_WEIGHTS_PATH}/pdp_plot.png",
    ) -> None:
        if self.model is None or self.X_test is None:
            msg = "Model or test data not loaded."
            raise ValueError(msg)

        _, ax = plt.subplots(figsize=(8, 4 * len(features)))
        PartialDependenceDisplay.from_estimator(
            self.model,
            self.X_test,
            features,
            ax=ax,
        )
        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()

    def get_rule_based_explanation(
    self,
    max_depth: int = 5,
    min_samples_leaf: int = 10,
    ) -> str:
        rule_exp_name = f"{Config.RF_MODEL_WEIGHTS_PATH}/rule_explanation.txt"
        if self.model is None or self.X_test is None:
            msg = "Model or test data not loaded."
            raise ValueError(msg)

        tree = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        tree.fit(self.X_test, self.model.predict(self.X_test))
        feature_names = list(self.X_test.columns)

        try:
            explanation = self.generate_rule_explanation_html(tree, feature_names)

            rule_exp_path = Path(rule_exp_name)
            with rule_exp_path.open("w", encoding="utf-8") as f:
                f.write(explanation)
        except Exception as e:
            logger.exception(f"Error generating explanation: {e}")
            # Fallback to textual export if error occurs
            from sklearn.tree import export_text
            rules = export_text(tree, feature_names=feature_names)
            return rules
        else:
            return explanation

    def generate_rule_explanation_html(self, tree, feature_names):
        def build_tree_structure(tree, feature_names, node_id=0):
            tree_ = tree.tree_

            feature = tree_.feature[node_id]
            threshold = tree_.threshold[node_id]
            left_child = tree_.children_left[node_id]
            right_child = tree_.children_right[node_id]
            value = tree_.value[node_id]

            if feature == -2:  # leaf node
                return {
                    "feature": None,
                    "threshold": None,
                    "left": None,
                    "right": None,
                    "value": float(value[0][0]),
                }
            return {
                "feature": feature_names[feature],
                "threshold": threshold,
                "left": build_tree_structure(tree, feature_names, left_child),
                "right": build_tree_structure(tree, feature_names, right_child),
                "value": None,
            }

        def explain_tree_to_html(node, depth=0):
            indent = "  " * depth
            if node["feature"] is None:
                return (
                    f"{indent}<li>Predict value: "
                    f"<strong>{node['value']:.3f}</strong></li>\n"
                )
            feature = escape(node["feature"])
            threshold = node["threshold"]
            html = (
                f"{indent}<li>Check if <strong>{feature}</strong> ≤ {threshold:.3f}:\n"
                f"{indent}<ul>\n"
            )
            # Left branch: condition true
            html += f"{indent}  <li><em>Yes</em> →\n"
            html += f"{indent}  <ul>\n"
            html += explain_tree_to_html(node["left"], depth + 3)
            html += f"{indent}  </ul>\n"
            html += f"{indent}  </li>\n"
            # Right branch: condition false
            html += f"{indent}  <li><em>No</em> →\n"
            html += f"{indent}  <ul>\n"
            html += explain_tree_to_html(node["right"], depth + 3)
            html += f"{indent}  </ul>\n"
            html += f"{indent}  </li>\n"
            html += f"{indent}</ul>\n"
            html += f"{indent}</li>\n"
            return html

        tree_dict = build_tree_structure(tree, feature_names)
        html = "<ul>\n"
        html += explain_tree_to_html(tree_dict)
        html += "</ul>"
        return html


if __name__ == "__main__":
    logger.info("check cache data=>")
    fi_save_name = f"{Config.RF_MODEL_WEIGHTS_PATH}/feature_importance.json"
    shap_exp_name = f"{Config.RF_MODEL_WEIGHTS_PATH}/shap_explanation.json"
    pdp_plot_name = f"{Config.RF_MODEL_WEIGHTS_PATH}/pdp_plot.png"
    rule_exp_name = f"{Config.RF_MODEL_WEIGHTS_PATH}/rule_explanation.txt"

    fi_path = Path(fi_save_name)
    shap_exp_path = Path(shap_exp_name)

    explainer = RandomForestExplainer()
    fi = explainer.get_feature_importance()
    shap_exp = explainer.get_shap_explanation([0])
    explainer.save_feature_importance_plot()
    explainer.save_shap_plot(0)

    top_features = sorted(
        zip(fi["feature_names"], fi["importances"]),
        key=lambda x: x[1],
        reverse=True,
    )[:4]
    top_feature_names = [f[0] for f in top_features]
    explainer.save_partial_dependence_plot(
        top_feature_names,
        out_path=pdp_plot_name,
    )

    # Save rule-based explanation
    rules = explainer.get_rule_based_explanation()

    # log results
    msg = f"   \tFeature Importance:   \n \t{json.dumps(fi, indent=2)}"
    logger.info(msg)

    msg = f"SHAP Explanation=> \n \t{json.dumps(shap_exp, indent=2)}"
    logger.info(msg)

    logger.info(rules)
