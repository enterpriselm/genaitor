from setuptools import setup, find_packages

setup(
    name="genaitor",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "google-generativeai>=0.3.0",
        "transformers>=4.30.0",
        "pdfplumber>=0.9.0",
        "python-pptx>=0.6.21",
        "python-docx>=0.8.11",
        "pandas>=1.5.0",
        "Pillow>=9.0.0",
        "pytesseract>=0.3.10",
        "moviepy>=1.0.3",
        "SpeechRecognition>=3.8.1",
        "pydub>=0.25.1",
        "langchain>=0.0.300",
        "langchain-community>=0.0.10",
        "requests>=2.31.0",
        "absl-py>=1.0.0",
        "grpcio>=1.50.0",
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "tenacity>=8.0.0"
    ]
) 