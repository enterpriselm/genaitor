from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from llama_agent import llama_agent_bp

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

app.register_blueprint(llama_agent_bp)

if __name__ == '__main__':
    app.run(debug=False, port=5000)