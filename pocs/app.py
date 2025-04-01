import streamlit as st
import subprocess
import sys

st.set_page_config(page_title="Genaitor", page_icon="🤖", layout="centered")

st.image("pocs/assets/logo.png", width=150)
st.title("Genaitor Examples")
st.write(
    "Genaitor is an innovative platform for creating AI agents and machine learning-based solutions. "
    "Explore the applications below to see examples of Genaitor's capabilities."
)

apps = {
    "🧩 Autism Assistant": "pocs/autism_assistant.py",
    "📐 CAD Analyst": "pocs/cad_analyst.py",
    "🚗 Car Buying Offer Generation": "pocs/configurator.py",
    "🔍 Dataset Anomaly Analysis": "pocs/anomalies_detection.py",
    "📧 E-mail Marketing Generator": "pocs/email_marketing_generator.py",
    "📊 PINN's Generator": "pocs/pinns.py",
    "📑 PPT and PDF Analyzer": "pocs/ppt_pdf_analyzer.py",
    "🛰️ Satellite Images Analyzer": "pocs/satellite_img_analyst.py",
    "🛡️ Websites Security Analyst": "pocs/security_analyst.py"
}

st.write("Select an application to run:")
selected_app = st.selectbox("Choose the app", list(apps.keys()))

if st.button("Run App"):
    app_path = apps[selected_app]
    st.write(f"Running {selected_app}...")
    subprocess.Popen(["streamlit", "run", app_path])

st.markdown("<p style='text-align: center; font-style: italic;'>Powered by Enterprise Learning Machines - 2025</p>", unsafe_allow_html=True)