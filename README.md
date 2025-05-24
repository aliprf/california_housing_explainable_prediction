# 🏠 California Housing Price Prediction

This project trains a Random Forest model to predict median house values using the California Housing dataset and provides a web-based UI for interaction.

---

## 📦 Install Dependencies

Install all required dependencies using [Pixi](https://pixi.sh):

```bash
pixi install -e prod
```

---

## 🏋️ Train the Model

Run the following command to train the Random Forest model:

```bash
pixi run train
```

This will:

* Load the dataset
* Train the model
* Save the model weights and evaluation data

---

## 🚀 Run the Server and Client

To start the backend server and the UI client:

```bash
pixi run up
```

Once running, the client will be accessible at:

👉 [http://127.0.0.1:7860/](http://127.0.0.1:7860/)

---

## 📁 Project Structure

```
.
🔽️ config.py                  # Centralized path & config values
🔽️ ml/
🔽️ ├── model/
🔽️ │   └── prediction.py      # Model training, saving, predicting
🔽️ ├── data_parser/
🔽️ │   └── data_parser.py     # Dataset loading and preprocessing
🔽️ └── explainability/        # SHAP/LIME model interpretability
🔽️ ui/                        # Web UI logic (e.g., Gradio app)
🔽️ weights/                   # Saved model and evaluation data
🔽️ pixi.toml                  # Pixi environment definition
🔽️ README.md
```

---

## ✅ Summary

* **Install dependencies**: `pixi install -e prod`
* **Train the model**: `pixi run train`
* **Start server & client**: `pixi run up`
* **Access UI**: [http://127.0.0.1:7860/](http://127.0.0.1:7860/)

---

