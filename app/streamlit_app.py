import streamlit as st
import requests

# Set up Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter text below to analyze its sentiment:")

# Text input
text = st.text_area("Input Text", "")

# Analyze button
if st.button("Analyze Sentiment"):
    if text.strip():
        # Call FastAPI endpoint for predictions
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",  # FastAPI endpoint
                json={"text": text}  # JSON payload
            )
            if response.status_code == 200:
                result = response.json()
                st.write(f"**Sentiment**: {result['sentiment']}")
                st.write(f"**Confidence**: {result['confidence']}")
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Error connecting to the prediction API: {str(e)}")
    else:
        st.error("Please enter some text to analyze.")