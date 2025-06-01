import gradio as gr
import requests

PREDICT_API_URL = "http://localhost:5000/predict"
INSIGHT_API_URL = "http://localhost:5000/model_insight"

# Custom dark theme using Base
dark_theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="gray",
).set(
    body_background_fill="#1e1e1e",
    body_text_color="#ffffff",
    button_primary_background_fill="#3b82f6",
    input_background_fill="#2d2d2d",
    block_background_fill="#2d2d2d",
    block_title_text_color="#ffffff",
)


def predict_house_value(
    MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude,
):
    payload = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude,
    }
    try:
        response = requests.post(PREDICT_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return f"💰 Predicted Median House Value: ${result['predicted_median_house_value']:.2f}"
    except requests.exceptions.RequestException as e:
        return f"❌ Request failed: {e}"
    except Exception as e:
        return f"⚠️ Error: {e!s}"


def load_insight():
    try:
        response = requests.post(INSIGHT_API_URL)
        response.raise_for_status()
        result = response.json()
        return (
            result["rules"],
            result["feature_importance_img"],
            result["shap_plot"],
            result["pdp_plot"],
        )
    except requests.exceptions.RequestException as e:
        return f"<p style='color:red;'>Failed to load insight: {e}</p>", None, None, None


def build_ui():
    with gr.Blocks(theme=dark_theme) as demo:
        with gr.Tab("Prediction"):
            gr.Markdown("### 🏠 California Housing Value Predictor")

            with gr.Row():
                MedInc = gr.Number(label="Median Income", value=5.0)
                HouseAge = gr.Number(label="House Age", value=20)

            with gr.Row():
                AveRooms = gr.Number(label="Average Rooms", value=6.0)
                AveBedrms = gr.Number(label="Average Bedrooms", value=1.0)

            with gr.Row():
                Population = gr.Number(label="Population", value=300)
                AveOccup = gr.Number(label="Average Occupants", value=3.0)

            with gr.Row():
                Latitude = gr.Number(label="Latitude", value=34.0)
                Longitude = gr.Number(label="Longitude", value=-118.0)

            predict_btn = gr.Button("Predict")
            prediction_output = gr.Textbox(label="Result", interactive=False)

            predict_btn.click(
                predict_house_value,
                inputs=[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude],
                outputs=prediction_output,
            )

        with gr.Tab("Prediction Insight"):
            gr.Markdown("## 📊 Model Insights and Explanation")

            load_btn = gr.Button("🔍 Load Insights")
            rules_html = gr.HTML(label="Decision Rules")

            fi_image = gr.Image(label="Feature Importance", show_label=False)
            shap_image = gr.Image(label="SHAP Explanation", show_label=False)
            pdp_image = gr.Image(label="Partial Dependence Plot", show_label=False)

            load_btn.click(
                load_insight,
                inputs=[],
                outputs=[rules_html, fi_image, shap_image, pdp_image],
            )

            gr.Markdown("### 🔎 Feature Importance")
            gr.Markdown("This plot shows how much each feature contributes to the model’s predictions.")

            gr.Markdown("### 🔎 SHAP Explanation")
            gr.Markdown("SHAP values explain the output of the model for a specific prediction.")

            gr.Markdown("### 🔎 Partial Dependence Plot")
            gr.Markdown("This shows the marginal effect of the top features on the predicted outcome.")

    return demo


def run_gradio():
    ui = build_ui()
    ui.launch(share=True)


if __name__ == "__main__":
    run_gradio()
