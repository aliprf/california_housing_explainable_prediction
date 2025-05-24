import gradio as gr
import requests

API_URL = "http://localhost:5000/predict"

def predict_house_value(MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude):
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
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return f"Predicted Median House Value: ${result['predicted_median_house_value']:.2f}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"
    except Exception as e:
        return f"Error: {str(e)}"

def build_ui():
    iface = gr.Interface(
        fn=predict_house_value,
        inputs=[
            gr.Number(label="Median Income (MedInc)", value=5.0),
            gr.Number(label="House Age", value=20),
            gr.Number(label="Average Rooms", value=6.0),
            gr.Number(label="Average Bedrooms", value=1.0),
            gr.Number(label="Population", value=300),
            gr.Number(label="Average Occupants", value=3.0),
            gr.Number(label="Latitude", value=34.0),
            gr.Number(label="Longitude", value=-118.0),
        ],
        outputs="text",
        title="California Housing Value Predictor",
        description="Enter housing and location details to estimate the median house value.",
    )
    return iface

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(share=True)

