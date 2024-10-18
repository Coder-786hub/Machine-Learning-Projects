import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Streamlit app configuration
st.set_page_config(page_title="Email Spam Checker", layout="centered")

# Heading
st.title("Email Spam Detector")

# Input for email content
email = st.text_area("Enter Email:", height=200)

# Button to check the email
if st.button("Check"):
    if email:
        # Vectorize and predict
        message_vectorized = vectorizer.transform([email])
        prediction = model.predict(message_vectorized)[0]
        result = "Spam" if prediction == 1 else "Non-Spam"
        # Display the result
        st.success(f"This email is classified as: {result}")
    else:
        st.warning("Please enter an email to check.")
