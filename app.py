# app.py

import streamlit as st
import joblib
import re
import string
import numpy as np

# Load saved model and vectorizer
model = joblib.load("veritasAI_model.pkl")
vectorizer = joblib.load("veritasAI_vectorizer.pkl")

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  # remove brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove links
    text = re.sub(r'<.*?>+', '', text)  # remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # remove punctuation
    text = re.sub(r'\n', ' ', text)  # remove line breaks
    text = re.sub(r'\w*\d\w*', '', text)  # remove words with numbers
    return text

# App Title
st.title("📰 VeritasAI - Fake News Detection")
st.markdown("**Paste a news article or social media text below and let VeritasAI detect if it's real or fake.**")

# User Input
user_input = st.text_area("🔎 Enter news article text here:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_text = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_input)[0]
        confidence = np.max(model.decision_function(vectorized_input))

        # Output
        label = "🟥 FAKE News" if prediction == 0 else "🟩 REAL News"
        st.subheader("🔍 Prediction Result:")
        st.success(label)
        st.info(f"Confidence Score: {confidence:.2f}")
