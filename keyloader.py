import os
import streamlit as st
from dotenv import load_dotenv

# Load .env locally
load_dotenv()

def get_secret(key: str):
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key)


