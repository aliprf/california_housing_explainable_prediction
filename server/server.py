import logging

from flask import Flask, jsonify, request, send_from_directory
from pydantic import ValidationError

from config import Config
from ml.common.schemas.california_housing_model import CaliforniaHousingModel
from ml.model.random_forest_explainer import RandomForestExplainer
from ml.model.random_forest_prediction import RandomForestPrediction

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load trained model once on startup
predictor = RandomForestPrediction()
predictor.load_model()


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)


@app.route("/model_insight", methods=["POST"])
def model_insight():
    pdp_plot_name = f"{Config.RF_MODEL_WEIGHTS_PATH}/pdp_plot.png"

    explainer = RandomForestExplainer()
    fi = explainer.get_feature_importance()
    explainer.get_shap_explanation([0])
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
    rules = explainer.get_rule_based_explanation()

    return jsonify({
        "rules": rules,
        "feature_importance_img": f"{Config.RF_MODEL_WEIGHTS_PATH}/feature_importance.png",
        "shap_plot": f"{Config.RF_MODEL_WEIGHTS_PATH}/shap_plot.png",
        "pdp_plot": f"{Config.RF_MODEL_WEIGHTS_PATH}/pdp_plot.png",
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()
        model_input = CaliforniaHousingModel(**input_data)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 422
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    prediction = predictor.predict(model_input)

    # Return prediction
    return jsonify({
        "predicted_median_house_value": prediction[0],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
