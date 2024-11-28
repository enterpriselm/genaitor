echo "Starting LLaVA model..."
./llava-v1.5-7b-q4.llamafile &
LAVA_PID=$!

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting Flask API..."
gunicorn -w 4 -b 0.0.0.0:8000 app:app --timeout 120

trap "kill $LAVA_PID" EXIT