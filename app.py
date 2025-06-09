from dotenv import load_dotenv
import os
from pathlib import Path
import streamlit as st
import requests

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("GROQ_API_KEY")

# App title
st.title("Chat with LLaMA 3 on Groq AI")
st.markdown("Built with Streamlit + Groq API")

# User input
prompt = st.text_area("Enter your prompt:", height=150)

# Send to Groq
if st.button("Generate"):
    if not prompt:
        st.warning("Please enter a prompt.")
    else:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama3-70b-8192",  # or other supported Groq models
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 256
        }

        with st.spinner("Generating..."):
            res = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers, json=payload
            )

        if res.status_code == 200:
            result = res.json()["choices"][0]["message"]["content"]
            st.success("Response:")
            st.write(result)
        else:
            st.error(f"Error {res.status_code}: {res.text}")
