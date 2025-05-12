import streamlit as st
import joblib

# Load the model
try:
    model = joblib.load('spam_classifier_model.pkl')
    feature_extraction = joblib.load('feature_extractor.pkl')
except Exception as e:
    st.error("Error loading the model. Please check if the required files exist.")
    st.stop()

# Page design
st.title("📧 Email Spam Detector")
st.write("This app helps you check if an email is spam or not")

# Model Performance Section
st.sidebar.title("📊 Model Performance")
st.sidebar.write("Model performance metrics on test data:")
st.sidebar.metric("Model Accuracy", "98.62%")

performance_details = """
Performance Details:
- Precision (Spam): 98%
- Recall (Spam): 99%
- F1-score (Spam): 99%
- Precision (Normal): 99%
- Recall (Normal): 98%
- F1-score (Normal): 99%
"""
st.sidebar.text(performance_details)

# Text input box
email_text = st.text_area(
    "Enter your email text here:",
    height=150,
    placeholder="Type or paste email content here..."
)

# Check button
if st.button("Check Email", type="primary"):
    if email_text.strip() == "":
        st.warning("Please enter email text first")
    else:
        with st.spinner("Checking..."):
            # Prediction
            email_features = feature_extraction.transform([email_text])
            prediction = model.predict(email_features)[0]
            probability = model.predict_proba(email_features)[0]

            # Show result
            if prediction == 1:
                st.success("✅ This is a NORMAL email")
                st.write(f"Confidence: {probability[1]*100:.0f}%")
            else:
                st.error("⚠️ This is a SPAM email")
                st.write(f"Confidence: {probability[0]*100:.0f}%")
