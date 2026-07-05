import os

print("Current Directory:", os.getcwd())
print("Model Exists:", os.path.exists("models/model.pkl"))

from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and feature names
model = pickle.load(open("models/model.pkl", "rb"))
features = pickle.load(open("models/features.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html", prediction=None)


@app.route("/predict", methods=["POST"])
def predict():

    # ===========================
    # Get Form Data
    # ===========================

    airline = request.form["Airline"]
    source = request.form["Source"]
    destination = request.form["Destination"]

    journey_date = request.form["Journey_Date"]
    dep_time = request.form["Dep_Time"]
    arrival_time = request.form["Arrival_Time"]

    duration_hours = int(request.form["Duration_Hours"])
    duration_minutes = int(request.form["Duration_Minutes"])

    total_stops = int(request.form["Total_Stops"])

    # ===========================
    # Feature Engineering
    # ===========================

    journey_date = pd.to_datetime(journey_date)

    journey_day = journey_date.day
    journey_month = journey_date.month

    dep_time = pd.to_datetime(dep_time)

    dep_hour = dep_time.hour
    dep_min = dep_time.minute

    arrival_time = pd.to_datetime(arrival_time)

    arrival_hour = arrival_time.hour
    arrival_min = arrival_time.minute

    # ===========================
    # Create DataFrame
    # ===========================

    data = pd.DataFrame({
        "Total_Stops": [total_stops],
        "Journey_Day": [journey_day],
        "Journey_Month": [journey_month],
        "Dep_Hour": [dep_hour],
        "Dep_Min": [dep_min],
        "Arrival_Hour": [arrival_hour],
        "Arrival_Min": [arrival_min],
        "Duration_Hours": [duration_hours],
        "Duration_Minutes": [duration_minutes],
        "Airline": [airline],
        "Source": [source],
        "Destination": [destination]
    })

    # ===========================
    # One Hot Encoding
    # ===========================

    data = pd.get_dummies(data)

    # ===========================
    # Add Missing Columns
    # ===========================

    for col in features:
        if col not in data.columns:
            data[col] = 0

    # Arrange Columns
    data = data[features]
    # ===========================
    # Prediction
    # ===========================

    prediction = model.predict(data)[0]
    prediction = int(round(prediction))

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)