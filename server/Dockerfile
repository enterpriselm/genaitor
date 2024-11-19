# Base Image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Run LLaVA model and Flask API
CMD ./llava-v1.5-7b-q4.llamafile & gunicorn -w 4 -b 0.0.0.0:5000 app:app