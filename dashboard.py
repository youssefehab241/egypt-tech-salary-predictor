import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Egypt Tech Salary Predictor",
    page_icon="💼",
    layout="centered"
)

MODEL_PATH = Path("salary_prediction_model/final_salary_model.pkl")

POSITIONS = [
    "back end engineer",
    "front end engineer",
    "full stack engineer",
    "data/ai engineer",
    "software testing engineer",
    "mobile engineer",
    "devops engineer",
    "embedded engineer",
    "technical support engineer",
    "cybersecurity engineer",
    "ui/ux designer"
]

POSITION_LABELS = {
    "back end engineer": "Back End Engineer",
    "front end engineer": "Front End Engineer",
    "full stack engineer": "Full Stack Engineer",
    "data/ai engineer": "Data / AI Engineer",
    "software testing engineer": "Software Testing Engineer",
    "mobile engineer": "Mobile Engineer",
    "devops engineer": "DevOps Engineer",
    "embedded engineer": "Embedded Engineer",
    "technical support engineer": "Technical Support Engineer",
    "cybersecurity engineer": "Cybersecurity Engineer",
    "ui/ux designer": "UI / UX Designer"
}


def load_model():
    return joblib.load(MODEL_PATH)


def predict_salary(model, job_title, experience_years):
    input_df = pd.DataFrame({
        "job_title": [job_title],
        "experience_years": [experience_years]
    })
    prediction = model.predict(input_df)[0]
    return max(0, float(prediction))


def get_smoothed_salary(model, job_title, target_experience):
    current_max = 0.0

    for exp in range(target_experience + 1):
        pred = predict_salary(model, job_title, exp)
        if pred > current_max:
            current_max = pred

    return current_max


st.title("Egypt Tech Salary Predictor")
st.write("Predict the expected monthly salary in Egypt based on job position and years of experience.")

if not MODEL_PATH.exists():
    st.error("Model file not found.")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

selected_position = st.selectbox(
    "Select your position",
    options=POSITIONS,
    format_func=lambda x: POSITION_LABELS[x]
)

experience_years = st.slider(
    "Years of experience",
    min_value=0,
    max_value=10,
    value=1,
    step=1
)

if st.button("Predict Salary"):
    try:
        predicted_salary = get_smoothed_salary(
            model,
            selected_position,
            experience_years
        )

        st.success("Prediction generated successfully")
        st.metric(
            "Estimated Monthly Salary (EGP)",
            f"{predicted_salary:,.0f} EGP"
        )

        st.write(
            f"Estimated salary for {POSITION_LABELS[selected_position]} "
            f"with {experience_years} years of experience is about "
            f"{predicted_salary:,.0f} EGP per month."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption("This is an estimated salary based on historical data.")