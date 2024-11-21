echo "Starting LLaVA model..."
./llava-v1.5-7b-q4.llamafile &
LAVA_PID=$!

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting Flask API..."
gunicorn -b 0.0.0.0:5000 app:app

trap "kill $LAVA_PID" EXIT