from flask import Flask
from llama_agent import llama_agent_bp

app = Flask(__name__)

app.register_blueprint(llama_agent_bp)

if __name__ == '__main__':
    app.run(debug=False, port=5000)