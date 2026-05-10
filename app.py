import streamlit as st
import joblib
import numpy as np

# Load saved objects
model = joblib.load("models\Fuel_Prediction_Model_Final.pkl")
scaler = joblib.load("models\scaler.pkl")
le = joblib.load("models\label_encoder.pkl")

# ---------------- UI ---------------- #
st.set_page_config(page_title="Fuel Type Predictor", page_icon="⛽", layout="centered")

st.title("⛽ Fuel Type Prediction App")
st.write("Enter vehicle details to predict the fuel type")

st.markdown("---")

# ---------------- INPUT ---------------- #
st.subheader("🚗 Vehicle Specifications")

mileage = st.text_input("Mileage (km/l)", placeholder="e.g. 16.5")
engine = st.text_input("Engine (CC)", placeholder="e.g. 1498")
power = st.text_input("Power (BHP)", placeholder="e.g. 98.96")

st.markdown("---")

# ---------------- PREDICTION ---------------- #
if st.button("🔮 Predict Fuel Type"):

    try:
        # convert input
        input_data = np.array([[float(mileage), float(engine), float(power)]])

        # scale input (IMPORTANT)
        input_scaled = scaler.transform(input_data)

        # predict (returns 0/1)
        prediction = model.predict(input_scaled)

        # decode back to label (Diesel / Petrol)
        fuel = le.inverse_transform(prediction)

        # ---------------- RESULT ---------------- #
        st.success(f"Predicted Fuel Type: {fuel[0]}")

        if fuel[0] == "Diesel":
            st.info("🟡 This vehicle is likely diesel-powered")
        elif fuel[0] == "Petrol":
            st.info("🔵 This vehicle is likely petrol-powered")
        else:
            st.info("⚪ Other fuel type detected")

    except Exception as e:
        st.error("⚠️ Please enter valid numeric values")

st.markdown("---")
st.caption("Built using Streamlit + Machine Learning 🚀")