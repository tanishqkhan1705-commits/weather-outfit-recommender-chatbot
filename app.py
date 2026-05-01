from flask import Flask, request, jsonify
import requests
import pickle
import pandas as pd

app = Flask(__name__)

API_KEY = "bf367f81e49da8cd9e0f1bcbd9f3b669"

# 🔥 LOAD MODELS
with open("outfit_model.pkl", "rb") as f:
    outfit_model = pickle.load(f)

with open("accessories_model.pkl", "rb") as f:
    accessories_model = pickle.load(f)

# ─────────────────────────────
# WEBHOOK
# ─────────────────────────────
@app.route("/webhook", methods=["POST"])
def webhook():

    req = request.get_json()
    params = req["queryResult"]["parameters"]

    print("PARAMS:", params)

    # 🔹 CITY
    city = params.get("geo-city")

    if isinstance(city, list):
        city = city[0]

    if not city:
        city = "Mumbai"

    city = str(city).strip().title()

    # 🔹 OCCASION
    occasion = params.get("occasion")

    if isinstance(occasion, list):
        occasion = occasion[0]

    if not occasion:
        occasion = "casual"

    occasion = str(occasion).strip().lower()

    print("CITY:", city)
    print("OCCASION:", occasion)

    # ─────────────────────────────
    # WEATHER API
    # ─────────────────────────────
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    res = requests.get(url)
    data = res.json()

    print("WEATHER DATA:", data)

    if str(data.get("cod")) != "200":
        return jsonify({
            "fulfillmentText": f"Weather not found for {city}"
        })

    temp = data["main"]["temp"]
    weather = data["weather"][0]["main"]

    # ─────────────────────────────
    # ML PREDICTION
    # ─────────────────────────────
    sample = pd.DataFrame([{
        "temp": temp,
        "weather": weather,
        "occasion": occasion
    }])

    predicted_outfit = outfit_model.predict(sample)[0]
    predicted_accessories = accessories_model.predict(sample)[0]

    # ─────────────────────────────
    # FINAL RESPONSE
    # ─────────────────────────────
    response_text = (
        f"{city}: {temp}°C, {weather}\n\n"
        f"👕 Outfit: {predicted_outfit}\n"
        f"👜 Accessories: {predicted_accessories}"
    )

    print("RESPONSE:", response_text)

    return jsonify({
        "fulfillmentText": response_text
    })

# ─────────────────────────────
# RUN
# ─────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)