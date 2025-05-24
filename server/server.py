from flask import Flask, request, jsonify
from pydantic import ValidationError
from ml.common.schemas.california_housing_model import CaliforniaHousingModel
import logging

from ml.model.random_forest_prediction import RandomForestPrediction



# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load trained model once on startup
predictor = RandomForestPrediction()
predictor.load_model()


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
        "predicted_median_house_value": prediction[0]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)